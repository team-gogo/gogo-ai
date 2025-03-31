import json
import logging
from pydantic import BaseModel
from typing import Optional

from aiokafka import AIOKafkaConsumer
from predict_model import predictor
from config import KAFKA_HOST, KAFKA_PORT
from event.producer import EventProducer
from schema.filter import filtered_result
    
events={
    'board_create':{
        'output':'boardId',
        'topic':'ai_board_filter'},
    'comment_create':{
        'output':'comment',
        'topic':'ai_comment_filter'},
    }

async def consume():
    event=list(events.keys())
    consumer = AIOKafkaConsumer(
        event[0], event[1], 
        bootstrap_servers=f'{KAFKA_HOST}:{KAFKA_PORT}',
    )

    await consumer.start()
    try:
        async for msg in consumer:
            data = json.loads(msg.value.decode('utf-8'))
            logging.info(f'Consume kafka data {msg.topic} value: {data}')
            if 'content' not in data:
                logging.error(f"Missing 'content' key in message: {data}")
                continue  # 'content' 키가 없으면 건너뜀
            logging.info(f"data: {data['content']}")
            model_output = await predictor(data['content'])
            logging.info(f'Predictor output: {model_output}')
            await EventProducer.create_event(
                topic=events[msg.topic]['topic'],
                key=data['id'],
                value=filtered_result(
                    id=data['id'],
                    **{events[msg.topic]['output']: model_output}
                ).dict(exclude_none=True),
            )
            

    except Exception as e:
        logging.exception(f'Kafka consume exception {str(e)}')
    finally:
        await consumer.stop()