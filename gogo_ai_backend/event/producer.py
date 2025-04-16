import json
import logging
from typing import Optional, Union

from aiokafka import AIOKafkaProducer
from pydantic import BaseModel

from config import get_kafka_host, get_kafka_port


class EventProducer:
    producer: Optional[AIOKafkaProducer] = None

    @staticmethod
    async def get_producer():
        if not EventProducer.producer:
            EventProducer.producer = AIOKafkaProducer(bootstrap_servers=f'{get_kafka_host()}:{get_kafka_port()}')
            await EventProducer.producer.start()
            logging.info(f'Producer is started.')
        return EventProducer.producer

    @staticmethod
    async def create_event(topic: str, key: str, value: Union[BaseModel, dict]):
        producer = await EventProducer.get_producer()

        if isinstance(value, BaseModel):
            value = value.dict(exclude_none=True)  

        await producer.send_and_wait(
            key=key.encode('utf-8'),
            topic=topic,
            value=json.dumps(value).encode('utf-8'), 
        )

        logging.info(f'Kafka producer Send {topic} value: {json.dumps(value)}') 
