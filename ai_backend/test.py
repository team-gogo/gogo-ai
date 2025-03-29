import uuid
from kafka import KafkaProducer
import json
producer = KafkaProducer(bootstrap_servers='127.0.0.1:9092')

event = {
	"id": str(uuid.uuid4()),
	"content": "와 이 시발 좆같은년아"
}

producer.send(
    'board_create',
    str(event).encode('utf-8')
)

