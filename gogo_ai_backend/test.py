import uuid
import json
from kafka import KafkaProducer
import random

producer = KafkaProducer(
    bootstrap_servers='127.0.0.1:9092',
    value_serializer=lambda v: json.dumps(v).encode('utf-8')  # JSON 자동 직렬화
)

event = {
    "id": str(uuid.uuid4()),
    "content": "안녕하세욥",
    "boardId": random.randint(1, 1000)
}

future = producer.send('board_create', event)

try:
    record_metadata = future.get(timeout=10)  # 메시지 전송 확인
    print(f"Message sent to topic {record_metadata.topic} partition {record_metadata.partition} at offset {record_metadata.offset}")
except Exception as e:
    print(f"Message delivery failed: {e}")


