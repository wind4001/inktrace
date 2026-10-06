#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""批量识别指定文件夹下所有图片, 输出逐张结果
用法: python batch_run.py [图片文件夹] [输出目录]  (默认 test/  output/)
"""
import sys
import time
from pathlib import Path

import recognize_phone as rp

BASE = Path(__file__).resolve().parent
FOLDER = Path(sys.argv[1]) if len(sys.argv) > 1 else (BASE / 'test')
OUT = BASE / (sys.argv[2] if len(sys.argv) > 2 else 'output')
if not FOLDER.exists():
    raise SystemExit(f'文件夹不存在: {FOLDER}')

print('加载模型...', flush=True)
det, rec = rp.make_ocr()
llm = rp.make_llm()

results = []
for img_path in sorted(p for p in FOLDER.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png')):
    t0 = time.time()
    try:
        r = rp.process_image(img_path, OUT, det, rec, llm, use_tiling=True)
        results.append((img_path.name, r))
        print(f"{img_path.name}: phone={r['phone'] or '(空)'} "
              f"review={r['needs_review']} regex={r['matches_regex']} "
              f"({time.time()-t0:.0f}s) | {r.get('reason','')}", flush=True)
    except Exception as e:
        results.append((img_path.name, None))
        print(f"{img_path.name}: ERROR {e}", flush=True)

print('\n=== 汇总 ===')
for name, r in results:
    if r is None:
        print(f'{name}: 识别失败')
    else:
        print(f'{name}: {r["phone"] or "(未识别)"} | review={r["needs_review"]} | regex={r["matches_regex"]}')
