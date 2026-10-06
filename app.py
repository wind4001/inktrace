#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
发票手写手机号识别 - Web 服务
运行: .venv/Scripts/python.exe app.py
浏览器打开: http://127.0.0.1:5000
"""
import threading
import uuid
from pathlib import Path

from flask import Flask, jsonify, render_template, request, send_from_directory

import recognize_phone as rp

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_DIR = BASE_DIR / 'uploads'
OUTPUT_DIR = BASE_DIR / 'output'
UPLOAD_DIR.mkdir(exist_ok=True)

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 80 * 1024 * 1024  # 单张上限 80MB

# 一次性加载 OCR 模型 + LLM 客户端(首次启动较慢, 之后复用)
print('加载 PaddleOCR 模型与 LLM 客户端(首次启动需数十秒)...')
DET, REC = rp.make_ocr()
LLM = rp.make_llm()
print('模型加载完成, 服务就绪')

JOBS = {}


def _run_job(job_id, img_path):
    try:
        result = rp.process_image(img_path, OUTPUT_DIR, DET, REC, LLM, use_tiling=True)
        JOBS[job_id] = {'status': 'done', 'result': result}
    except Exception as e:
        JOBS[job_id] = {'status': 'error', 'error': str(e)}


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files.get('file')
    if not f or not f.filename:
        return jsonify({'error': '未选择文件'}), 400
    job_id = uuid.uuid4().hex[:12]
    ext = Path(f.filename).suffix.lower()
    if ext not in ('.jpg', '.jpeg', '.png', '.bmp', '.webp', '.tif', '.tiff'):
        ext = '.jpg'
    filename = f'{job_id}{ext}'
    filepath = UPLOAD_DIR / filename
    f.save(filepath)
    JOBS[job_id] = {'status': 'processing', 'filename': f.filename}
    threading.Thread(target=_run_job, args=(job_id, str(filepath)), daemon=True).start()
    return jsonify({'job_id': job_id})


@app.route('/api/status/<job_id>')
def status(job_id):
    job = JOBS.get(job_id)
    if not job:
        return jsonify({'error': '任务不存在'}), 404
    return jsonify(job)


@app.route('/output/<path:filename>')
def output_file(filename):
    return send_from_directory(OUTPUT_DIR, filename)


if __name__ == '__main__':
    print('请用浏览器打开 http://127.0.0.1:5000')
    app.run(host='127.0.0.1', port=5000, threaded=True)
