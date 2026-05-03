"""Prometheus 메트릭 정의 — 한 곳에 모아 관리.

라벨 카디널리티를 낮게 유지하기 위해 model_revision은 short SHA(8자) 사용.
"""
from prometheus_client import Counter, Gauge, Histogram

PREDICT_LATENCY = Histogram(
    "profanity_predict_latency_seconds",
    "Profanity inference latency",
    labelnames=("model_revision",),
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
)

PREDICTION_TOTAL = Counter(
    "profanity_prediction_total",
    "Predictions by label",
    labelnames=("label", "model_revision"),
)

MODEL_LOAD_DURATION = Histogram(
    "profanity_model_load_duration_seconds",
    "Model + tokenizer load duration",
    buckets=(1, 5, 10, 30, 60, 120, 300, 600),
)

MODEL_INFO = Gauge(
    "profanity_model_info",
    "Loaded model provenance (value is always 1; metadata in labels)",
    labelnames=("model", "revision", "tokenizer", "tokenizer_revision"),
)

KAFKA_MESSAGE_TOTAL = Counter(
    "profanity_kafka_message_total",
    "Kafka messages consumed by topic and outcome",
    labelnames=("topic", "status"),
)
