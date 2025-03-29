import json
import logging
from pydantic import BaseModel
from typing import Optional

from aiokafka import AIOKafkaConsumer
from predict_model import predictor
from config import KAFKA_HOST, KAFKA_PORT
from event.producer import EventProducer

class filtered_result(BaseModel):
    id: str
    boardId: Optional[int] = None
    comment: Optional[int] = None
    
events={
    'board_create':'boardId', 'comment_create':'comment'
}

async def consume():
    event=list(events.keys())
    consumer = AIOKafkaConsumer(
        event[0], event[1], 
        bootstrap_servers=f'{KAFKA_HOST}:{KAFKA_PORT}'
    )

    await consumer.start()
    try:
        async for msg in consumer:
            data = json.loads(msg.value.decode('utf-8'))
            logging.info(f'Consume kafka data {msg.topic} value: {data}')
            model_output = await predictor(data['content'])
            logging.info(f'Predictor output: {model_output}')
            await EventProducer.create_event(
                topic=msg.topic,
                key=data['id'],
                value=filtered_result(
                    id=data['id'],
                    **{events[msg.topic]: model_output}
                ),
            )

    except Exception as e:
        logging.exception(f'Kafka consume exception {str(e)}')
    finally:
        await consumer.stop()