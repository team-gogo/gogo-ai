import json
import logging
from typing import Optional

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

from config import KAFKA_HOST, KAFKA_PORT


class EventProducer:
    producer: Optional[AIOKafkaProducer] = None

    @staticmethod
    async def get_producer():
        if not EventProducer.producer:
            EventProducer.producer = AIOKafkaProducer(bootstrap_servers=f'{KAFKA_HOST}:{KAFKA_PORT}')
            await EventProducer.producer.start()
            logging.info(f'Producer is started.')
        return EventProducer.producer

    @staticmethod
    async def create_event(topic: str, key: str, value: BaseModel):
        producer = await EventProducer.get_producer()
        await producer.send_and_wait(
            key=key.encode('utf-8'),
            topic=topic,
            value=json.dumps(value.dict()).encode('utf-8'),
        )
        logging.info(f'Kafka producer Send {topic} value: {value}')