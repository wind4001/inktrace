#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核心配置与纯逻辑测试。

覆盖范围: LLM 配置解析、缓存键生成、提示词清洁度、源码卫生。
不覆盖: 真实 OCR(需 PaddleOCR 模型) 与真实大模型调用 —— 那两项需要
完整环境与 API Key, 请按 README 手动验证。

运行(无需 pytest):
    python tests/test_core.py

也可以用 pytest:
    pytest tests/
"""
import os
import re
import sys
import tempfile
import types
from pathlib import Path

# recognize_phone 在模块层 import cv2。本文件只测配置与纯逻辑分支, 不触碰任何
# 图像路径, 所以 cv2 缺失时用一个空模块占位, 让测试能在未装重型依赖的轻量环境
# (如 CI) 里跑。装了 cv2 的正常环境走真实模块。
for _m in ('cv2',):
    try:
        __import__(_m)
    except ImportError:
        sys.modules[_m] = types.ModuleType(_m)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np                     # noqa: E402
import recognize_phone as rp           # noqa: E402

SRC = (ROOT / 'recognize_phone.py').read_text(encoding='utf-8')
# 提示词里有意保留的假号(中国惯例, 等价于美国的 555-0100), 是合法示例值
DUMMY_PHONE = '13800138000'
LLM_ENV = ('LLM_API_KEY', 'LLM_BASE_URL', 'LLM_MODEL')


def _clear_llm_env():
    for k in LLM_ENV:
        os.environ.pop(k, None)


def _catch_exit(fn):
    """跑 fn, 返回 SystemExit 的消息(没有则返回 None)"""
    try:
        fn()
    except SystemExit as e:
        return str(e)
    return None


# --------------------------- 缓存键 ---------------------------

def test_content_key_is_stable_for_same_content():
    a = np.zeros((10, 10, 3), np.uint8)
    assert rp.content_key(a) == rp.content_key(a)


def test_content_key_differs_for_different_content():
    a = np.zeros((10, 10, 3), np.uint8)
    b = np.ones((10, 10, 3), np.uint8)
    assert rp.content_key(a) != rp.content_key(b)


def test_content_key_is_ten_chars():
    assert len(rp.content_key(np.zeros((4, 4, 3), np.uint8))) == 10


def test_content_key_does_not_depend_on_filename():
    """回归: 缓存键必须只看内容。同名不同图若共用键, 第二张会静默读到第一张的结果。"""
    img = np.zeros((8, 8, 3), np.uint8)
    cropped = img[:4]
    assert rp.content_key(img) != rp.content_key(cropped)


# --------------------------- LLM 配置 ---------------------------

def test_missing_api_key_exits_with_clear_message():
    _clear_llm_env()
    msg = _catch_exit(rp.make_llm)
    assert msg is not None and 'LLM_API_KEY' in msg


def test_missing_model_exits_with_clear_message():
    _clear_llm_env()
    os.environ['LLM_API_KEY'] = 'sk-test-placeholder'
    msg = _catch_exit(rp.make_llm)
    assert msg is not None and 'LLM_MODEL' in msg


def test_base_url_defaults_when_omitted():
    _clear_llm_env()
    os.environ['LLM_API_KEY'] = 'sk-test-placeholder'
    os.environ['LLM_MODEL'] = 'any-model'
    assert 'api.deepseek.com' in str(rp.make_llm().base_url)


def test_base_url_override_is_honoured():
    """指向本地 Ollama 之类, 是离线使用的路子"""
    _clear_llm_env()
    os.environ.update(LLM_API_KEY='sk-x', LLM_MODEL='any-model',
                      LLM_BASE_URL='http://127.0.0.1:11434/v1')
    assert '11434' in str(rp.make_llm().base_url)


def test_empty_base_url_falls_back_to_default():
    _clear_llm_env()
    os.environ.update(LLM_API_KEY='sk-x', LLM_MODEL='any-model', LLM_BASE_URL='')
    assert 'api.deepseek.com' in str(rp.make_llm().base_url)


def test_llm_model_reads_env():
    _clear_llm_env()
    os.environ['LLM_MODEL'] = 'qwen-plus'
    assert rp.llm_model() == 'qwen-plus'


def test_llm_json_passes_configured_model():
    """请求里的 model 必须来自 .env, 不能是代码里写死的"""
    _clear_llm_env()
    os.environ['LLM_MODEL'] = 'qwen-plus'

    class _Completions:
        def __init__(self):
            self.kwargs = None

        def create(self, **kw):
            self.kwargs = kw
            msg = types.SimpleNamespace(content='{"phone":"13800138000"}')
            return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    fake = types.SimpleNamespace(
        chat=types.SimpleNamespace(completions=_Completions()))
    out = rp.llm_json(fake, 'prompt')
    assert out == {'phone': DUMMY_PHONE}
    assert fake.chat.completions.kwargs['model'] == 'qwen-plus'


# --------------------------- 磁盘缓存 ---------------------------

def test_cache_hit_returns_stored_lines():
    tmp = tempfile.mkdtemp()
    rp.OCR_CACHE_DIR = Path(tmp)
    img = np.zeros((8, 8, 3), np.uint8)
    key = rp.content_key(img)
    (Path(tmp) / f'receipt_{key}.json').write_text(
        '{"image":"receipt.jpg","W":100,"H":200,"generated_at":"test","lines":'
        '[{"box":[[0,0],[10,0],[10,5],[0,5]],"text":"13800138000","conf":0.9,'
        '"cx":5,"cy":2,"hw":false}]}', encoding='utf-8')

    lines, h, w = rp.get_all_lines(img, 'receipt.jpg', None, None, cache_key=key)
    assert len(lines) == 1 and lines[0]['text'] == DUMMY_PHONE
    assert isinstance(lines[0]['box'], np.ndarray)
    assert (h, w) == (200, 100)


# --------------------------- 提示词清洁度 ---------------------------
# 这些是回归护栏: 提示词里曾硬编码过真实手机号与真实票据金额, 必须防止再被写回。

def test_prompt_has_no_real_phone_numbers():
    found = [n for n in re.findall(r'\b1[3-9]\d{9}\b', SRC) if n != DUMMY_PHONE]
    assert not found, f'提示词/注释里出现手机号形态的数字: {found}'


def test_prompt_uses_amount_placeholder():
    assert '<金额>' in SRC, '金额提示词应使用 <金额> 占位, 而不是具体数值'


def test_prompt_has_no_hardcoded_money_literals():
    assert not re.search(r'-\d+\.\d{2}', SRC), '不应有减项形态的具体金额'
    assert not re.search(r'\b\d{4}\.\d{2}\b', SRC), '不应有具体金额字面量'


# --------------------------- 源码卫生 ---------------------------

def test_no_provider_env_var_leftover():
    assert 'DEEPSEEK_' not in SRC, '旧的环境变量名应已全部换成 LLM_*'


def test_no_hardcoded_model_id():
    """模型名必须来自 .env, 代码里不能写死任何一家的"""
    assert 'deepseek-v4' not in SRC and 'deepseek-chat' not in SRC


def test_no_hardcoded_absolute_paths():
    src_files = [ROOT / f for f in
                 ('recognize_phone.py', 'app.py', 'batch_run.py')]
    for f in src_files:
        text = f.read_text(encoding='utf-8')
        hits = re.findall(r'[A-Za-z]:[\\/][A-Za-z0-9_]', text)
        assert not hits, f'{f.name} 含硬编码盘符路径: {hits}'


# --------------------------- 运行器 ---------------------------

def _main():
    tests = [(n, f) for n, f in sorted(globals().items())
             if n.startswith('test_') and callable(f)]
    passed, failed = [], []
    for name, fn in tests:
        try:
            fn()
        except Exception as e:
            failed.append(name)
            print(f'  FAIL  {name}\n        {type(e).__name__}: {e}')
        else:
            passed.append(name)
            print(f'  PASS  {name}')
    print('\n' + '=' * 56)
    print(f'  PASS {len(passed)}   FAIL {len(failed)}')
    if failed:
        print('  FAILED: ' + ', '.join(failed))
    print('=' * 56)
    return 1 if failed else 0


if __name__ == '__main__':
    _clear_llm_env()
    sys.exit(_main())
