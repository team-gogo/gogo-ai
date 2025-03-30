from fastapi import FastAPI
import logging
import asyncio
import uvicorn
from middleware import LoggingMiddleware
from event.consumer import consume
from contextlib import asynccontextmanager

@asynccontextmanager
async def lifespan(app: FastAPI):
    asyncio.create_task(consume())  # 비동기 태스크 실행
    yield


app = FastAPI(lifespan=lifespan)

logging.basicConfig(level=logging.INFO)
app.add_middleware(LoggingMiddleware)

@app.get("/ai/health")
async def root():
    return 'GOGO Ai Service OK'

if __name__ == '__main__':
    try:
        # init_eureka()
        # logging.info('init eureka')
        uvicorn.run(app, host='0.0.0.0', port=8087, log_level='info', access_log=False)
    except Exception as e:
        print(e)
        exit(1)