#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发票/小票手写手机号识别
流程:
  1) PaddleOCR 全页 + 分块检测识别所有文本行(含坐标)
     分块是为了解决大图内部缩放导致的小手写漏检(如小票顶部的手写号码)
  2) 墨迹连通域查找检测漏掉的手写候选(倾斜/潦草手写), 多角度尽力识别
  3) LLM 从所有文本行中判别手写手机号(任何 OpenAI 兼容协议的服务均可)
  4) 手机号正则校验 + needs_review 标记(防止 LLM 幻觉补全)

用法:
    python recognize_phone.py <图片路径> [--output <目录>] [--no-tiling]

配置: 复制 .env.example 为 .env, 填入 LLM_API_KEY / LLM_BASE_URL / LLM_MODEL
"""

import argparse
import hashlib
import json
import os
import re
import time
from pathlib import Path

import cv2
import numpy as np
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / '.env')

PHONE_RE = re.compile(r'^1[3-9]\d{9}$')
TILE = 1800          # 分块尺寸
TILE_OVERLAP = 300   # 分块重叠

# LLM 走 OpenAI 兼容协议, 任何提供该协议的服务都能用(DeepSeek/通义/Kimi/GLM/OpenAI/
# Ollama/vLLM 等). 具体用哪个由 .env 的 LLM_* 决定, 代码里不写死任何模型名.
DEFAULT_BASE_URL = 'https://api.deepseek.com'


def content_key(img):
    """按图片内容取短哈希, 用于缓存/输出文件名。
    不用文件名做键: 不同目录下的同名图片(如 a/1.jpg 与 b/1.jpg)会共用缓存,
    第二张图会直接读到第一张的 OCR 结果——静默错判。内容相同则复用缓存(有意为之)。"""
    return hashlib.sha1(np.ascontiguousarray(img).tobytes()).hexdigest()[:10]


# ------------------------- 图像 / 坐标工具 -------------------------

def rectify_line(img, poly):
    """把四角框透视矫正为水平文本条"""
    poly = np.asarray(poly, dtype=np.float32)
    tl, tr, br, bl = poly
    w = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    h = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    w, h = max(int(round(w)), 1), max(int(round(h)), 1)
    dst = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
    M = cv2.getPerspectiveTransform(poly, dst)
    return cv2.warpPerspective(img, M, (w, h))


def upscale(crop, factor=2):
    if factor <= 1:
        return crop
    return cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)


def imread_unicode(path):
    """兼容中文/特殊字符路径的读图(Windows 上 cv2.imread 不支持非 ASCII 路径)"""
    data = np.fromfile(str(path), dtype=np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def expand_poly(poly, factor=1.2):
    poly = np.asarray(poly, dtype=np.float32)
    cx, cy = poly[:, 0].mean(), poly[:, 1].mean()
    return (poly - [cx, cy]) * factor + [cx, cy]


def normalize_box(poly, W, H):
    xs, ys = poly[:, 0], poly[:, 1]
    return (min(xs) / W * 100, min(ys) / H * 100, max(xs) / W * 100, max(ys) / H * 100)


def region_tag(cx, cy):
    v = '上' if cy < 33 else ('中' if cy < 66 else '下')
    h = '左' if cx < 33 else ('中' if cx < 66 else '右')
    return v + h


# ------------------------- PaddleOCR -------------------------

def make_ocr():
    """模型可通过 .env 配置(OCR_DET_MODEL/OCR_REC_MODEL), 默认 PP-OCRv6 medium
    回退 v5 时: OCR_DET_MODEL=PP-OCRv5_server_det  OCR_REC_MODEL=PP-OCRv5_server_rec
    enable_mkldnn=False: v6 模型在 paddle 3.3.1 的 oneDNN 路径有
    ConvertPirAttribute2RuntimeAttribute bug, 需走纯 paddle 路径
    """
    from paddleocr import TextDetection, TextRecognition
    det_name = os.getenv('OCR_DET_MODEL', 'PP-OCRv6_medium_det')
    rec_name = os.getenv('OCR_REC_MODEL', 'PP-OCRv6_medium_rec')
    det = TextDetection(model_name=det_name, device='cpu', enable_mkldnn=False)
    rec = TextRecognition(model_name=rec_name, device='cpu', enable_mkldnn=False)
    return det, rec


def _item(result, name):
    try:
        return result[name]
    except (KeyError, TypeError):
        return getattr(result, name)


def extract_polys(result):
    r = result[0]
    for key in ('dt_polys', 'dt_boxes', 'rec_polys', 'polygons', 'boxes'):
        try:
            arr = np.array(_item(r, key), dtype=np.float32)
            if arr.ndim == 3 and arr.shape[1:] == (4, 2):
                return arr
        except (KeyError, TypeError, AttributeError):
            continue
    # 空结果/无法解析: 返回空数组
    return np.zeros((0, 4, 2), dtype=np.float32)


def detect_boxes(det, img):
    return extract_polys(det.predict(img))


def merge_boxes(boxes, min_dist=30):
    """按中心距离去重(全页与分块结果重叠)"""
    kept = []
    for b in boxes:
        bc = b.mean(axis=0)
        dup = any(np.linalg.norm(k.mean(axis=0) - bc) < min_dist for k in kept)
        if not dup:
            kept.append(np.asarray(b, dtype=np.float32))
    return kept


def full_detect_boxes(det, img, use_tiling=True):
    """全页 + 分块 检测, 返回合并去重后的全图坐标四角框"""
    H, W = img.shape[:2]
    all_boxes = list(detect_boxes(det, img))
    if use_tiling:
        step = TILE - TILE_OVERLAP
        for y0 in range(0, H, step):
            for x0 in range(0, W, step):
                y1, x1 = min(y0 + TILE, H), min(x0 + TILE, W)
                if y1 - y0 < 120 or x1 - x0 < 120:
                    continue
                sub = img[y0:y1, x0:x1]
                try:
                    for p in detect_boxes(det, sub):
                        all_boxes.append(p + np.array([x0, y0], dtype=np.float32))
                except Exception:
                    pass
    return merge_boxes(all_boxes)


def rec_lines(rec, img, boxes):
    """对四角框逐行矫正+识别, 返回 lines"""
    lines = []
    for box in boxes:
        box = np.asarray(box, dtype=np.float32)
        crop = rectify_line(img, box)
        try:
            r = rec.predict(upscale(crop, 2))[0]
            text = _item(r, 'rec_text')
            conf = float(_item(r, 'rec_score'))
        except Exception:
            continue
        if text.strip():
            lines.append({
                'box': box, 'text': text, 'conf': conf,
                'cx': float(box[:, 0].mean()), 'cy': float(box[:, 1].mean()),
                'hw': False,
            })
    return lines


def find_handwriting(det, rec, img, known_lines):
    """查找检测漏掉的手写手机号:
    1) 把已检测的印刷文字区域涂白
    2) 对涂白图上残留墨迹做连通域聚类, 定位候选手写区域
    3) 对每个候选区域裁剪, 用 det+rec 在区域尺度上重新识别(解决整页缩放漏检)
    """
    H, W = img.shape[:2]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    bw = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                               cv2.THRESH_BINARY_INV, 31, 15)
    # 涂白印刷文字区域
    mask = np.zeros((H, W), np.uint8)
    for ln in known_lines:
        cv2.fillPoly(mask, [ln['box'].astype(np.int32)], 255)
    mask = cv2.dilate(mask, np.ones((24, 24), np.uint8))
    masked_img = img.copy()
    masked_img[mask > 0] = 255

    # 涂白后残留墨迹 = 候选(手写/印章/条码)
    ink = cv2.bitwise_and(bw, cv2.bitwise_not(mask))
    n, labels, stats, cents = cv2.connectedComponentsWithStats(ink, 8)
    comps = []
    for i in range(1, n):
        x, y, w, h, area = stats[i]
        if 12 < h < 300 and 6 < w < 900 and area > 80:
            comps.append([x, y, w, h, area])

    # 按 x 排序, 宽松聚类(可能含少量印章/残留, 交给 det 精修)
    comps.sort(key=lambda c: c[0])
    clusters = []
    for c in comps:
        placed = False
        for cl in clusters:
            last = cl['items'][-1]
            x_gap = c[0] - (last[0] + last[2])
            y_drift = abs(c[1] - last[1])
            max_h = max(c[3], last[3])
            if x_gap < 3.0 * max_h and y_drift < 2.5 * max_h:
                cl['items'].append(c)
                placed = True
                break
        if not placed:
            clusters.append({'items': [c]})

    hw_lines = []
    seen = set()
    for cl in clusters:
        its = cl['items']
        if len(its) < 3:
            continue
        # 按 y 排序, 按 y-间隙把高簇切成横带(避免印刷残留把簇撑高)
        its = sorted(its, key=lambda c: c[1])
        med_h = float(np.median([c[3] for c in its]))
        bands = [[its[0]]]
        for prev, cur in zip(its, its[1:]):
            if cur[1] - (prev[1] + prev[3]) > 1.5 * med_h:
                bands.append([cur])
            else:
                bands[-1].append(cur)
        for grp in bands:
            if len(grp) < 3:
                continue
            x0 = min(c[0] for c in grp)
            y0 = min(c[1] for c in grp)
            x1 = max(c[0] + c[2] for c in grp)
            y1 = max(c[1] + c[3] for c in grp)
            w, h = x1 - x0, y1 - y0
            if w < 200 or h < 10 or h > 300:
                continue
            # 从涂白图上裁剪, 在区域尺度跑 det(避免整页缩放的漏检)
            pad = 20
            crop = masked_img[max(y0 - pad, 0):min(y1 + pad, H), max(x0 - pad, 0):min(x1 + pad, W)]
            try:
                polys = detect_boxes(det, crop)
            except Exception:
                continue
            for p in polys:
                cimg = rectify_line(crop, p)
                try:
                    r = rec.predict(upscale(cimg, 2))[0]
                    t, s = _item(r, 'rec_text'), float(_item(r, 'rec_score'))
                except Exception:
                    continue
                if not t.strip():
                    continue
                nd = sum(ch.isdigit() for ch in t)
                if nd < 7:
                    continue
                box = p + np.array([x0 - pad, y0 - pad])
                key = (int(box[:, 0].mean()), int(box[:, 1].mean()))
                if key in seen:
                    continue
                seen.add(key)
                hw_lines.append({
                    'box': box.astype(np.float32), 'text': t, 'conf': float(s),
                    'cx': float(box[:, 0].mean()), 'cy': float(box[:, 1].mean()),
                    'hw': True,
                })
    return hw_lines


# ------------------------- LLM (OpenAI 兼容协议) -------------------------

def make_llm():
    """构造 OpenAI 兼容协议的客户端, 服务商由 .env 决定。
    LLM_BASE_URL 省略时回退 DeepSeek 官方端点; LLM_MODEL 必填——
   各家 model id 不同, 猜错只会换来一个 400, 不如让它明确报错。"""
    key = os.environ.get('LLM_API_KEY', '').strip()
    if not key or '在这里' in key or '在此' in key:
        raise SystemExit('请复制 .env.example 为 .env, 并填入 LLM_API_KEY')
    model = os.environ.get('LLM_MODEL', '').strip()
    if not model:
        raise SystemExit('请在 .env 中配置 LLM_MODEL (写法见 .env.example 中的示例)')
    from openai import OpenAI
    base_url = os.environ.get('LLM_BASE_URL', '').strip() or DEFAULT_BASE_URL
    return OpenAI(api_key=key, base_url=base_url)


def llm_model():
    """取 .env 中配置的模型名; 供 llm_json 调用"""
    return os.environ.get('LLM_MODEL', '').strip()


def llm_json(llm, user, retries=4):
    """调用 LLM 并解析 JSON; 对临时错误(过载/限流/连接/5xx)自动重试退避"""
    from openai import (APIConnectionError, APITimeoutError,
                        InternalServerError, RateLimitError)
    for attempt in range(retries):
        try:
            resp = llm.chat.completions.create(
                model=llm_model(),
                messages=[
                    {'role': 'system', 'content': '你是发票信息核对助手。只输出合法的 JSON 对象，不输出任何其他文字、注释或 markdown。'},
                    {'role': 'user', 'content': user},
                ],
                temperature=0.0,
            )
            text = resp.choices[0].message.content.strip()
            text = re.sub(r'^```(json)?\s*', '', text)
            text = re.sub(r'\s*```$', '', text)
            start, end = text.find('{'), text.rfind('}')
            if start == -1 or end == -1:
                raise RuntimeError(f'LLM 未返回 JSON: {text[:300]}')
            return json.loads(text[start:end + 1])
        except (APIConnectionError, APITimeoutError, InternalServerError, RateLimitError):
            if attempt == retries - 1:
                raise
            print(f'      LLM 临时错误, {2 ** (attempt + 1)}s 后重试 ({attempt + 1}/{retries}) ...', flush=True)
            time.sleep(2 ** (attempt + 1))


OCR_CACHE_DIR = BASE_DIR / 'ocr_cache'


def get_all_lines(img, img_path, det, rec, use_tiling=True, cache_key=None):
    """带磁盘缓存的整页OCR(检测+识别+墨迹查找): 同一张图只识别一次, 之后从缓存读取.
    返回 (all_lines, H, W)。缓存文件: ocr_cache/<图名>_<内容哈希>.json"""
    img_path = Path(img_path)
    key = cache_key or content_key(img)
    cache_file = OCR_CACHE_DIR / f'{img_path.stem}_{key}.json'
    if cache_file.exists():
        data = json.loads(cache_file.read_text(encoding='utf-8'))
        all_lines = []
        for l in data['lines']:
            d = dict(l)
            d['box'] = np.array(d['box'], dtype=np.float32)
            all_lines.append(d)
        print(f'      [缓存] {cache_file.name}: {len(all_lines)} 行(生成于 {data.get("generated_at", "?")})')
        return all_lines, int(data['H']), int(data['W'])

    H, W = img.shape[:2]
    print('[2/5] PaddleOCR 检测(全页 + 分块) ...')
    boxes = full_detect_boxes(det, img, use_tiling=use_tiling)
    print(f'      检测到 {len(boxes)} 个文本框')
    print('[3/5] 识别文本行 ...')
    lines = rec_lines(rec, img, boxes)
    lines.sort(key=lambda l: (l['cy'], l['cx']))
    print(f'      识别 {len(lines)} 行')
    print('      墨迹查找手写候选 ...')
    hw_lines = find_handwriting(det, rec, img, lines)
    all_lines = lines + hw_lines
    all_lines.sort(key=lambda l: (l['cy'], l['cx']))
    if hw_lines:
        print(f'      发现 {len(hw_lines)} 条手写候选: ' +
              ', '.join(f'{l["text"]!r}({l["conf"]:.2f})' for l in hw_lines))
    else:
        print('      未发现独立手写候选')
    OCR_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps({
        'image': str(img_path), 'W': int(W), 'H': int(H),
        'generated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        'lines': [{k: (v.tolist() if k == 'box' else v) for k, v in l.items()} for l in all_lines],
    }, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f'      已缓存到 {cache_file.name}')
    return all_lines, H, W


def build_discriminate_prompt(lines, W, H):
    rows = []
    for i, ln in enumerate(lines):
        x1, y1, x2, y2 = normalize_box(ln['box'], W, H)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        hw_mark = ' [手写候选]' if ln.get('hw') else ''
        rows.append(f'[{i+1}] "{ln["text"]}" conf:{ln["conf"]:.2f} '
                    f'box:({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f}) 分区:{region_tag(cx, cy)}{hw_mark}')
    return (
        '以下是一张发票/小票整页 OCR 识别的所有文本行，坐标已归一化为页面百分比(0-100)，'
        'x1,y1=左上角，x2,y2=右下角；行号按阅读顺序(上→下、左→右)排列，相邻行号=空间相邻。\n\n'
        '发票背景知识：\n'
        '- 印刷体电话位于"销售方信息"等区域，行内带"电话/地址"等中文标签，box 规整、置信度通常>0.9。\n'
        '- 要找的是【手写】的手机号：独立一行纯数字(11位、以1开头、第二位为3-9)，写在空白/备注/签名区，'
        '无中文标签，置信度往往较低，数字可能有个别识别误差。\n'
        '- 带 [手写候选] 标记的行是通过墨迹分析找到的疑似手写内容(常规检测漏检的倾斜/潦草手写)，'
        '应优先从这些行中判断手写手机号。\n'
        '- 【重要】同一行手写手机号可能被检测成 2-3 个相邻/重叠的片段行(位于同一区域、文本全是数字，'
        '例如 \'13800\' 和 \'138000\' 两个相邻数字行能拼成 \'13800138000\')。\n\n'
        '任务：找出【手写】的那个手机号。\n'
        '只输出一个 JSON 对象：\n'
        '{"phone":"数字串","line_index":整数(对应上方行号),"reason":"一句话理由",'
        '"ocr_text":"该行OCR原始文本","needs_review":bool}\n'
        '规则：\n'
        '1. 优先选11位、以1开头、第二位为3-9的数字串。\n'
        '2. 若相邻/重叠的数字行拼接后正好是 11 位合规手机号，就输出拼接结果，'
        'line_index 填第一段的行号，reason 里注明拼接了哪些行。\n'
        '3. 【重要】如果手写候选行的 OCR 只读出部分数字(如8位)且无法通过拼接补全，'
        '就如实输出这些数字，并把 needs_review 设为 true；绝不推测/补全缺失的数字。\n'
        '4. needs_review: 若最终 phone 直接来自 OCR 原文(含片段拼接但未改任何数字)→false；'
        '其余情况→true。\n'
        '5. 若完全无法确定，phone 填空字符串""，needs_review 填 true。\n\n'
        '文本行列表：\n' + '\n'.join(rows)
    )


def build_adjudicate_prompt(candidates):
    rows = [f'[{k}] "{t}" conf:{c:.2f}' for k, t, c in candidates]
    return (
        '下面是同一行手写手机号经过不同方式 OCR 得到的候选结果：\n' + '\n'.join(rows) + '\n'
        '结合手机号格式约束(11位、以1开头、第二位为3-9)，判断哪个候选最接近真实手机号，'
        '也可以综合多个候选推断真实号码。\n'
        '【重要】如果所有候选都是部分数字(少于11位)，就输出最长的那个，needs_review 填 true，'
        '绝不补全缺失数字。\n'
        '只输出 JSON：{"phone":"数字串","source":"A/B/C/综合","reason":"一句话理由","needs_review":bool}'
    )


def build_amount_prompt(lines, W, H):
    """从整页文本行中判别【实付金额】(不含优惠券/优惠减免的最终付款金额)"""
    rows = []
    for i, ln in enumerate(lines):
        x1, y1, x2, y2 = normalize_box(ln['box'], W, H)
        rows.append(f'[{i+1}] "{ln["text"]}" conf:{ln["conf"]:.2f} '
                    f'box:({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})')
    return (
        '以下是一张发票/小票整页 OCR 识别的所有文本行，坐标已归一化为页面百分比(0-100)，'
        'x1,y1=左上角，x2,y2=右下角。\n\n'
        '任务：找出这张小票的【实付金额】——消费者最终实际支付的金额，格式为数字(可带小数，如 123.45、1234)。\n\n'
        '判断规则(按优先级)：\n'
        '1. 优先取"支付金额"：带"支付/付款/现金/实付/支付总计/微信/支付宝"等支付方式标签旁的金额'
        '(如"微信支付：<金额>")——那是消费者实际付款。\n'
        '2. 若小票有【优惠券/优惠金额】抵扣(如"优惠券-<金额>"、"优惠金额-<金额>"、单独一行"<金额>"券面额)，'
        '实付金额 = 应付/合计 - 优惠抵扣额。注意：即使"实收金额/应付金额"标签写的是未扣券金额(如原价)，'
        '只要存在优惠券抵扣，也要减去优惠券(得到抵扣后的值)。"发票金额/开票金额"通常按实付开具，可作交叉验证。\n'
        '3. 【折扣金额/折扣】标签(如"折扣-<金额>")表示已享受的折扣，合计金额已是折扣后的最终值，'
        '实付取支付金额或合计，【不要】再从合计里减去折扣。\n'
        '4. 若以上都无，取"合计/总计/总额/总金额/实收/应付"标签的金额。\n'
        '5. 排除：商品单价、小计、找零、退款、积分兑换、赠送金额等中间/非支付项。\n'
        '6. 金额一般在标签同一行(如"实付金额：<金额>")或紧邻的下一行。\n'
        '7. 若整页没有任何明确的金额汇总(OCR只识别出"合计"等标签但无数值)，amount 输出空字符串""，'
        'needs_review 填 true，绝不编造金额。\n'
        '8. 若金额数值缺失但标签存在(如只有"总额"二字无数值)，请把标签所在行号填入 label_index，'
        '以便程序对该区域放大重读。\n'
        '只输出一个 JSON 对象：\n'
        '{"amount":"数字或小数","line_index":整数(金额数值所在行号,取不到填0),'
        '"label_index":整数(金额标签行如"实付/合计/总额/支付总计"所在行号,找不到填0),'
        '"reason":"一句话理由","needs_review":bool}'
        '\n\n文本行列表：\n' + '\n'.join(rows)
    )


def _clean_amount(s):
    return (s or '').strip().replace('￥', '').replace('¥', '').replace(',', '').strip()


def build_date_prompt(lines, W, H):
    """从整页文本行中判别【开票日期】"""
    rows = []
    for i, ln in enumerate(lines):
        x1, y1, x2, y2 = normalize_box(ln['box'], W, H)
        rows.append(f'[{i+1}] "{ln["text"]}" conf:{ln["conf"]:.2f} '
                    f'box:({x1:.0f},{y1:.0f})-({x2:.0f},{y2:.0f})')
    return (
        '以下是一张发票/小票整页 OCR 识别的所有文本行，坐标已归一化为页面百分比(0-100)，'
        'x1,y1=左上角，x2,y2=右下角。\n\n'
        '任务：找出这张小票的【开票日期】——即这张小票开具/打印的日期，'
        '通常格式如 2026-07-10、2026/07/10、2026年7月10日，可能在日期旁带时间(如 2026-07-10 10:30)。\n\n'
        '判断规则：\n'
        '1. 优先找带"开票日期/开票时间/日期/交易时间/打印时间"等标签的日期行。\n'
        '2. 否则取小票顶部或单据信息区域的日期(小票的交易/开具日期通常就在那里)。\n'
        '3. 排除：商品生产日期/有效期/到期日、会员生日、保修期等无关日期。\n'
        '4. 若日期行含时间(如 "2026-07-10 10:30:22")，只取日期部分。\n'
        '5. 输出统一为 YYYY-MM-DD 格式(如 2026-07-10)。年份统一用四位。\n'
        '6. 若整页没有任何日期，date 输出空字符串""，needs_review 填 true，绝不编造。\n'
        '只输出一个 JSON 对象：\n'
        '{"date":"YYYY-MM-DD","line_index":整数(日期所在行号,取不到填0),"reason":"一句话理由","needs_review":bool}'
        '\n\n文本行列表：\n' + '\n'.join(rows)
    )


def re_read_amount_region(img, det, rec, label_line):
    """对金额标签行附近的横条区域放大重读, 返回候选文本列表"""
    box = label_line['box']
    y0 = int(box[:, 1].min()); y1 = int(box[:, 1].max())
    h = max(y1 - y0, 20)
    y0 = max(0, y0 - 2 * h); y1 = min(img.shape[0], y1 + 2 * h)
    region = img[y0:y1, :]
    cands = []
    big = upscale(region, 2)
    try:
        r = rec.predict(big)[0]
        t = _item(r, 'rec_text'); s = float(_item(r, 'rec_score'))
        if t.strip():
            cands.append(('整条放大2x', t, s))
    except Exception:
        pass
    try:
        rboxes = full_detect_boxes(det, region, use_tiling=False)
        for l in rec_lines(rec, region, rboxes):
            if any(c.isdigit() for c in l['text']):
                cands.append((f'区域行@y{l["cy"]:.0f}', l['text'], l['conf']))
    except Exception:
        pass
    return cands


def build_amount_adjudicate_prompt(region_cands, initial_note):
    rows = '\n'.join(f'[{k}] "{t}" conf:{c:.2f}' for k, t, c in region_cands)
    return (
        f'以下是发票/小票金额标签附近区域放大重读得到的文本(含 OCR 候选):\n{rows}\n'
        f'初次判断说明: {initial_note}\n'
        '任务：结合上述候选，确定这张小票的【实付金额】(不含优惠券/优惠减免)。\n'
        '金额格式为数字(可带两位小数)。\n'
        '只输出一个 JSON 对象：\n'
        '{"amount":"数字","reason":"一句话理由","needs_review":bool}\n'
        '若仍无法确定，amount 输出空字符串""，needs_review 填 true。'
    )


# ------------------------- 重叠行交叉核对 -------------------------
# 同一手写区域可能被检测成多个框(全页框/前段框/后段框), 逐位对齐后投票.
# 冲突位用"同图模板匹配"裁决: 同一个人笔迹稳定, 用图中已确认位的字形做模板.
# 注: 依赖 v6 也适用的通用逻辑, 与 OCR 模型无关.

def _norm_char_img(crop, seg):
    x0, x1 = max(seg[0], 0), min(seg[1], crop.shape[1])
    ch = crop[:, x0:x1]
    gray = cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (40, 80), interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 255.0


def _norm_corr(a, b):
    a = a - a.mean(); b = b - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / max(d, 1e-9))


def _segment_chars(crop, N):
    """列投影切字符; 粘连段按二分搜索字符宽细分, 使总段数精确等于 N.
    返回 N 个 (x0,x1) 段, 无法满足则返回 None"""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    colsum = bw.sum(axis=0)
    in_gap, segs = True, []
    for x, v in enumerate(colsum):
        if v > 0 and in_gap:
            segs.append([x]); in_gap = False
        elif v == 0 and not in_gap:
            segs[-1].append(x); in_gap = True
    if not in_gap:
        segs[-1].append(len(colsum) - 1)
    merged = []
    for s in segs:
        if merged and s[0] - merged[-1][1] < 8:   # 断笔合并
            merged[-1][1] = s[1]
        else:
            merged.append(s)
    segs = merged
    if not segs or len(segs) > N:
        return None
    widths = [s[1] - s[0] for s in segs]
    total = segs[-1][1]

    def count_at(cw):
        return sum(max(1, round(w / cw)) for w in widths)

    lo, hi = total / N * 0.4, total / N * 2.5
    best_cw, best_diff = lo, abs(count_at(lo) - N)
    for _ in range(50):
        mid = (lo + hi) / 2
        cnt = count_at(mid)
        d = abs(cnt - N)
        if d < best_diff:
            best_cw, best_diff = mid, d
        if cnt > N:      # 段分太多 → 字符宽应增大
            lo = mid
        elif cnt < N:    # 分太少 → 字符宽应减小
            hi = mid
        else:
            best_cw, best_diff = mid, 0
            break
    if best_diff > 0:
        return None
    final = []
    for s in segs:
        w = s[1] - s[0]
        k = max(1, round(w / best_cw))
        for j in range(k):
            x0 = s[0] + j * w / k
            x1 = s[0] + (j + 1) * w / k
            final.append([int(x0), int(x1)])
    if len(final) != N:
        return None
    return final


def cross_validate_lines(all_lines, img):
    """对同一手写区域的重叠数字行逐位对齐投票, 冲突位用同图模板匹配裁决.
    返回 (notes_text, consensus_phone) 或 (None, None)"""
    dlines = []
    for i, ln in enumerate(all_lines):
        digits = re.sub(r'\D', '', ln['text'])
        if len(digits) >= 5:
            dlines.append({'line': ln, 'digits': digits, 'seq': i + 1})

    def overlap(a, b):
        """真正的"同一手写文本多框": 同起点不同长/同终点不同长/一方显著包含另一方.
        排除同一打印行的不同字段(日期+单号等并排文本)."""
        ab, bb = a['line']['box'], b['line']['box']
        ax0, ay0 = ab[:, 0].min(), ab[:, 1].min()
        ax1, ay1 = ab[:, 0].max(), ab[:, 1].max()
        bx0, by0 = bb[:, 0].min(), bb[:, 1].min()
        bx1, by1 = bb[:, 0].max(), bb[:, 1].max()
        cy_a, cy_b = (ay0 + ay1) / 2, (by0 + by1) / 2
        if abs(cy_a - cy_b) > 0.8 * (ay1 - ay0):   # y 中心不接近
            return False
        if ax1 <= bx0 or bx1 <= ax0:               # x 不重叠
            return False
        na, nb = len(a['digits']), len(b['digits'])
        wa, wb = (ax1 - ax0) / max(na, 1), (bx1 - bx0) / max(nb, 1)
        mcw = max(wa, wb)
        if abs(ax0 - bx0) < 0.5 * mcw:             # 同起点(前段框/全行框)
            return True
        if abs(ax1 - bx1) < 0.5 * mcw:             # 同终点(后段框/全行框)
            return True
        wa_abs, wb_abs = ax1 - ax0, bx1 - bx0
        if wa_abs > 1.6 * wb_abs:                  # a 显著包含 b
            return ax0 <= bx0 + 0.25 * wb_abs and bx1 <= ax1 + 0.25 * wb_abs
        if wb_abs > 1.6 * wa_abs:                  # b 显著包含 a
            return bx0 <= ax0 + 0.25 * wa_abs and ax1 <= bx1 + 0.25 * wa_abs
        return False

    # 并查集分组(允许经中间框桥接)
    parent = list(range(len(dlines)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for i in range(len(dlines)):
        for j in range(i + 1, len(dlines)):
            if overlap(dlines[i], dlines[j]):
                union(i, j)
    group_map = {}
    for i in range(len(dlines)):
        group_map.setdefault(find(i), []).append(dlines[i])
    groups = list(group_map.values())

    final_notes, final_phone = [], None
    for g in groups:
        if len(g) < 2:
            continue
        g = sorted(g, key=lambda l: -len(l['digits']))
        base, N = g[0], len(g[0]['digits'])
        base_box = base['line']['box']
        bx0, bx1 = base_box[:, 0].min(), base_box[:, 0].max()
        char_w = max((bx1 - bx0) / N, 1e-6)
        votes = [[] for _ in range(N)]
        for ln in g:
            m = len(ln['digits'])
            lx0 = ln['line']['box'][:, 0].min()
            # 子串优先对齐(片段框是基准框的子串, 如后段框'38000'是'13800138000'后缀);
            # 几何估计在字符宽不均时可能错 1 位(05 因此产生假冲突)。
            off = base['digits'].find(ln['digits'])
            if off < 0:
                off = round((lx0 - bx0) / char_w)
            for i in range(m):
                pos = off + i
                if 0 <= pos < N:
                    votes[pos].append((ln['digits'][i], ln['seq']))
        conflicts = [i for i, v in enumerate(votes) if len({x[0] for x in v}) > 1]
        if not conflicts:
            continue

        # 同图模板库: 投票一致的位
        base_crop = rectify_line(img, base_box)
        segs = _segment_chars(base_crop, N)
        if segs is None:
            continue
        templates = {}
        for i, v in enumerate(votes):
            vals = {x[0] for x in v}
            if len(vals) == 1:
                templates.setdefault(vals.pop(), []).append(_norm_char_img(base_crop, segs[i]))

        consensus = list(base['digits'])
        notes = [f'行{"/".join(str(gl["seq"]) for gl in g)} 覆盖同一手写区域, '
                 f'{N}位中仅以下位存在跨行冲突:']
        for i in conflicts:
            cand = sorted({x[0] for x in votes[i]})
            unk = _norm_char_img(base_crop, segs[i])
            scores = {}
            for label, tpls in templates.items():
                scores[label] = max(_norm_corr(unk, t) for t in tpls)
            ranked = sorted(scores.items(), key=lambda kv: -kv[1])
            if ranked and len(ranked) > 1 and (ranked[0][1] - ranked[1][1]) > 0.3:
                winner = ranked[0][0]
                consensus[i] = winner
                notes.append(f'  第{i+1}位: 各行读{" vs ".join(cand)}, '
                             f'同图模板匹配判为{winner!r}'
                             f'(corr {ranked[0][1]:.2f} vs {ranked[1][1]:.2f}, 模板来自本图已确认位)')
            else:
                notes.append(f'  第{i+1}位: 各行读{" vs ".join(cand)}, '
                             f'模板匹配证据不足, 保留基准值{base["digits"][i]!r}')
        if any(consensus[i] != base['digits'][i] for i in conflicts):
            p = ''.join(consensus)
            notes.append(f'  综合建议号码: {p} (其中冲突位为模板推断, 建议人工复核)')
            if final_phone is None and PHONE_RE.match(p):   # 仅采纳合规手机号
                final_phone = p
                final_notes = '\n'.join(notes)
    if final_phone is None:
        return None, None
    return final_notes, final_phone


# ------------------------- 主流程 -------------------------

def process_image(img_path, out_dir, det, rec, llm, use_tiling=True):
    """核心流程: 检测+识别+LLM判别, 返回结果dict并写输出文件(CLI和Web共用)"""
    img_path = Path(img_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f'[1/5] 加载图片: {img_path}')
    img = imread_unicode(img_path)
    if img is None:
        raise SystemExit(f'图片读取失败: {img_path}')

    key = content_key(img)
    all_lines, H, W = get_all_lines(img, img_path, det, rec,
                                   use_tiling=use_tiling, cache_key=key)

    print('[4/5] LLM 判别手写手机号 ...')
    # 重叠行交叉核对: 同一手写区域的多框逐位投票, 冲突位同图模板匹配
    cv_notes, cv_phone = cross_validate_lines(all_lines, img)
    prompt = build_discriminate_prompt(all_lines, W, H)
    if cv_notes:
        prompt += ('\n\n【重叠行交叉核对】(同一手写区域被检测成多个框, 代码已逐位对齐):\n'
                   + cv_notes +
                   '\n注意: 若采纳上述"综合建议号码"(其冲突位为同图模板匹配推断), '
                   '则 needs_review 必须为 true, 并在 reason 中注明第几位是模板推断。')
    disc = llm_json(llm, prompt)
    idx = disc.get('line_index')
    chosen = all_lines[idx - 1] if idx and 1 <= idx <= len(all_lines) else None

    result = {
        'image': str(img_path),
        'phone': re.sub(r'\D', '', disc.get('phone', '') or ''),
        'line_index': idx,
        'reason': disc.get('reason', ''),
        'ocr_text': disc.get('ocr_text', ''),
        'needs_review': bool(disc.get('needs_review', True)),
        'matches_regex': bool(PHONE_RE.match(re.sub(r'\D', '', disc.get('phone', '') or ''))),
        'is_handwriting_candidate': bool(chosen and chosen.get('hw')),
    }
    if cv_notes and cv_phone:
        # 交叉核对有强证据改判(多框投票 + 同图模板匹配 corr 差>0.3) → 代码层直接采用,
        # 不依赖 LLM 是否采纳(LLM 对冲突信息采纳不稳定), 并强制人工复核。
        result['phone'] = cv_phone
        result['needs_review'] = True
        result['matches_regex'] = True
        result['cross_validated'] = cv_notes

    # 实付金额识别(不含优惠券/优惠减免)
    print('      识别实付金额 ...')
    amt_disc = llm_json(llm, build_amount_prompt(all_lines, W, H))
    amt_text = _clean_amount(amt_disc.get('amount'))
    result['amount'] = amt_text
    result['amount_reason'] = amt_disc.get('reason', '')
    result['amount_needs_review'] = bool(amt_disc.get('needs_review', True))
    label_idx = amt_disc.get('label_index') or amt_disc.get('line_index') or 0
    if (not amt_text or result['amount_needs_review']) and label_idx and 1 <= label_idx <= len(all_lines):
        # 金额缺失/存疑 → 对标签行附近区域放大重读后综合判断
        print(f'      金额区域重读(标签行{label_idx}) ...')
        cands = re_read_amount_region(img, det, rec, all_lines[label_idx - 1])
        if cands:
            adj = llm_json(llm, build_amount_adjudicate_prompt(cands, amt_disc.get('reason', '')))
            adj_amt = _clean_amount(adj.get('amount'))
            if adj_amt:
                amt_text = adj_amt
                result['amount'] = adj_amt
                result['amount_reason'] = (f'{adj.get("reason", "")} | 区域重读: '
                                           + '; '.join(f'{t}' for _, t, _ in cands))
                result['amount_needs_review'] = bool(adj.get('needs_review', True))
            result['amount_region_candidates'] = [t for _, t, _ in cands]

    # 开票日期识别
    print('      识别开票日期 ...')
    date_disc = llm_json(llm, build_date_prompt(all_lines, W, H))
    result['date'] = (date_disc.get('date') or '').strip()
    result['date_reason'] = date_disc.get('reason', '')
    result['date_needs_review'] = bool(date_disc.get('needs_review', True))

    if chosen is not None and not result['matches_regex']:
        # 二次精读: 初次结果不是完整合规手机号时才重读选定区域
        print('      二次精读选定区域 ...')
        base = rectify_line(img, expand_poly(chosen['box'], 1.2))
        gray = cv2.cvtColor(base, cv2.COLOR_BGR2GRAY)
        variants = [
            ('原图矫正放大', upscale(base, 3)),
            ('灰度放大', upscale(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 3)),
            ('CLAHE增强', upscale(cv2.cvtColor(
                cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(gray),
                cv2.COLOR_GRAY2BGR), 3)),
        ]
        candidates = [('初识', chosen['text'], chosen['conf'])]
        for name, vimg in variants:
            try:
                r = rec.predict(vimg)[0]
                t = _item(r, 'rec_text')
                s = float(_item(r, 'rec_score'))
                candidates.append((name, t, s))
            except Exception as e:
                print(f'      变体 {name} 识别失败: {e}')
        adjud = llm_json(llm, build_adjudicate_prompt(candidates))
        final_phone = re.sub(r'\D', '', adjud.get('phone', '') or '')
        result.update({
            'phone': final_phone,
            'reason': ' | '.join(x for x in (adjud.get('reason', ''), disc.get('reason', '')) if x),
            'needs_review': bool(adjud.get('needs_review', True)) or not PHONE_RE.match(final_phone),
            'matches_regex': bool(PHONE_RE.match(final_phone)),
            'candidates': candidates,
            'chosen_box': chosen['box'].round(1).tolist(),
            'is_handwriting_candidate': bool(chosen.get('hw')),
        })

    # 标注图
    vis = img.copy()
    for i, ln in enumerate(all_lines):
        color = (0, 0, 255) if chosen is not None and i == idx - 1 else (
            (0, 140, 255) if ln.get('hw') else (0, 180, 0))
        cv2.polylines(vis, [ln['box'].astype(np.int32)], True, color, 5)
        cv2.putText(vis, str(i + 1), (int(ln['box'][0][0]), int(ln['box'][0][1]) - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, color, 4)
    out_img = out_dir / f'{img_path.stem}_{key}_result.jpg'
    cv2.imencode('.jpg', vis)[1].tofile(str(out_img))

    out_json = out_dir / f'{img_path.stem}_{key}.json'
    # 先补齐字段再写盘, 让落盘 JSON 与返回值一致(文件里也带上自己的图/结果文件名)
    result['result_image'] = out_img.name
    result['json_file'] = out_json.name
    out_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    return result


def main():
    ap = argparse.ArgumentParser(description='发票/小票手写手机号识别')
    ap.add_argument('image', help='图片路径')
    ap.add_argument('--output', default=str(BASE_DIR / 'output'), help='输出目录')
    ap.add_argument('--no-tiling', action='store_true', help='关闭分块检测')
    args = ap.parse_args()

    img_path = Path(args.image)
    if not img_path.exists():
        raise SystemExit(f'图片不存在: {img_path}')
    out_dir = Path(args.output)

    print('加载 OCR 模型与 LLM 客户端 ...')
    det, rec = make_ocr()
    llm = make_llm()

    result = process_image(img_path, out_dir, det, rec, llm,
                           use_tiling=not args.no_tiling)

    print('\n=== 结果 ===')
    print(f'手写手机号: {result["phone"] or "(未识别出)"}')
    print(f'needs_review: {result["needs_review"]}   regex合法: {result["matches_regex"]}')
    print(f'手写候选标记: {result["is_handwriting_candidate"]}')
    print(f'理由: {result["reason"]}')
    print(f'标注图: {out_dir / result["result_image"]}')
    print(f'JSON: {out_dir / result["json_file"]}')


if __name__ == '__main__':
    main()
