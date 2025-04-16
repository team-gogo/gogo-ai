import os

from dotenv import load_dotenv

load_dotenv('../.env', override=False)

# EUREKA_HOST = os.environ.get('EUREKA_HOST')
# EUREKA_PORT = os.environ.get('EUREKA_PORT')
# EUREKA_SERVICE_NAME = os.environ.get('EUREKA_SERVICE_NAME')
# INSTANCE_IP = os.environ.get('INSTANCE_IP')
# INSTANCE_HOST = os.environ.get('INSTANCE_HOST')
# INSTANCE_PORT = int(os.environ.get('INSTANCE_PORT'))

def get_kafka_host():
    return os.getenv('KAFKA_HOST')

def get_kafka_port():
    return os.getenv('KAFKA_PORT')