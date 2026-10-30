"""3社のBatch送信・回収と、返却形式の違いを扱う。"""
from __future__ import annotations

import json
import hashlib
import os
from pathlib import Path


def create_client(provider):
    """明示的なオンライン操作時だけSDKクライアントを作る。作成APIの自動再送はしない。"""
    if provider == 'openai':
        from openai import OpenAI
        return OpenAI(timeout=600, max_retries=0)
    if provider == 'anthropic':
        from anthropic import Anthropic
        return Anthropic(timeout=600, max_retries=0)
    if provider == 'google':
        from google import genai
        from google.genai import types
        return genai.Client(api_key=os.environ.get('GEMINI_API_KEY') or os.environ.get('GOOGLE_API_KEY'),
                            http_options=types.HttpOptions(timeout=600000,
                                retry_options=types.HttpRetryOptions(attempts=1)))
    raise ValueError(f'未対応provider: {provider}')


def submit_one(client, provider, path, model, endpoint='/v1/responses'):
    """JSONLを1ジョブとして送信し、回収に必要なIDを返す。"""
    if provider == 'openai':
        with path.open('rb') as f:
            uploaded = client.files.create(file=f, purpose='batch')
        job = client.batches.create(input_file_id=uploaded.id, endpoint=endpoint, completion_window='24h')
        return {'job_id': job.id, 'input_file_id': uploaded.id}
    if provider == 'anthropic':
        requests = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        job = client.messages.batches.create(requests=requests)
        return {'job_id': job.id}
    from google.genai import types
    uploaded = client.files.upload(file=str(path), config=types.UploadFileConfig(mime_type='jsonl'))
    identity = '-'.join(path.parts[-5:]).removesuffix('.jsonl')
    fingerprint = hashlib.sha256(path.read_bytes()).hexdigest()[:10]
    job = client.batches.create(model=model, src=uploaded.name,
                               config={'display_name': identity[:110] + '-' + fingerprint})
    return {'job_id': job.name, 'input_file_id': uploaded.name}


def get_job(client, provider, job_id):
    """1回だけ状態を取得し、終端状態かどうかと生メタデータを返す。"""
    if provider == 'openai':
        job = client.batches.retrieve(job_id)
        state = job.status
        terminal = state in {'completed', 'failed', 'expired', 'cancelled'}
    elif provider == 'anthropic':
        job = client.messages.batches.retrieve(job_id)
        state = job.processing_status
        terminal = state == 'ended'
    else:
        job = client.batches.get(name=job_id)
        state = job.state.name
        terminal = state in {'JOB_STATE_SUCCEEDED', 'JOB_STATE_FAILED', 'JOB_STATE_CANCELLED', 'JOB_STATE_EXPIRED'}
    return job, state, terminal


def download_results(client, provider, job):
    """成功と失敗の両方を回収する。欠落行は後段でexpectedキーと照合する。"""
    if provider == 'openai':
        rows = []
        for file_id in [job.output_file_id, job.error_file_id]:
            if file_id:
                rows.extend(json.loads(line) for line in client.files.content(file_id).text.splitlines() if line.strip())
        return rows
    if provider == 'anthropic':
        return [r.model_dump(mode='json') for r in client.messages.batches.results(job.id)]
    if job.dest and job.dest.file_name:
        content = client.files.download(file=job.dest.file_name).decode('utf-8')
        return [json.loads(line) for line in content.splitlines() if line.strip()]
    return []


def response_text(provider, row):
    """正常終了した最終テキストだけを取り出す。拒否・途中打切り・APIエラーは失敗とする。"""
    if row.get('error'):
        raise ValueError(f'API error: {row["error"]}')
    if provider == 'openai':
        response = row.get('response') or {}
        if response.get('status_code') != 200:
            raise ValueError(f'HTTP error: {response.get("status_code")}')
        body = response.get('body') or {}
        if body.get('status') != 'completed' or body.get('error'):
            raise ValueError(f'incomplete response: {body.get("status")}')
        parts = [part for item in body.get('output', []) if item.get('type') == 'message'
                 for part in item.get('content', [])]
        if any(p.get('type') == 'refusal' for p in parts):
            raise ValueError('refusal')
        text = ''.join(p.get('text', '') for p in parts if p.get('type') == 'output_text')
    elif provider == 'anthropic':
        result = row.get('result') or {}
        if result.get('type') != 'succeeded':
            raise ValueError(f'Batch result: {result.get("type")}')
        body = result['message']
        if body.get('stop_reason') != 'end_turn':
            raise ValueError(f'stop_reason: {body.get("stop_reason")}')
        text = ''.join(p.get('text', '') for p in body.get('content', []) if p.get('type') == 'text')
    else:
        body = row.get('response') or {}
        candidates = body.get('candidates') or []
        if len(candidates) != 1 or candidates[0].get('finishReason') != 'STOP':
            raise ValueError(f'incomplete/blocked candidate: {body.get("promptFeedback", candidates)}')
        text = ''.join(p.get('text', '') for p in candidates[0].get('content', {}).get('parts', []) if not p.get('thought'))
    if not text.strip():
        raise ValueError('empty text')
    return text, body.get('usage') or body.get('usageMetadata') or {}
