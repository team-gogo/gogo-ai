import json
import uuid
import time
import logging

from fastapi import Request, Response
from starlette.concurrency import iterate_in_threadpool
from starlette.middleware.base import BaseHTTPMiddleware

logger = logging.getLogger('GOGO-AI Logger')
no_logging_path = ['/ai/health','/favicon.ico']


class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        if request.url.path in no_logging_path:
            return await call_next(request)

        log_id = str(uuid.uuid4())
        start_time = time.time()

        request_body = await request.body()
        try:
            request_body_str = json.dumps(json.loads(request_body.decode('utf-8')), separators=(',', ':'))
        except json.decoder.JSONDecodeError:
            request_body_str = ''

        logger.info(
            f"Log-ID: {log_id}, "
            f"IP: {request.client.host}, "
            f"URI: {request.url.path}, "
            f"Http-Method: {request.method}, "
            f"Params: {json.dumps(dict(request.query_params), separators=(',', ':'))}, "
            f"Content-Type: {request.headers.get('content-type')}, "
            f"User-Agent: {request.headers.get('user-agent')}, "
            f"Request-Body: {request_body_str}"
        )

        response = await call_next(request)

        response_body = [chunk async for chunk in response.body_iterator]
        response.body_iterator = iterate_in_threadpool(iter(response_body))
        response_body_str = b"".join(response_body).decode('utf-8')

        process_time = round((time.time() - start_time) * 1000, 2)
        logger.info(
            f"Log-ID: {log_id}, "
            f"Status-Code: {response.status_code}, "
            f"Content-Type: {response.headers.get('content-type')}, "
            f"Response Time: {process_time}ms, "
            f"Response-Body: {json.dumps(json.loads(response_body_str), separators=(',', ':'))}"
        )

        return response