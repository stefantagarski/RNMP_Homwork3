import json
from kafka import KafkaConsumer

consumer = KafkaConsumer(
    "health_data_predicted",
    bootstrap_servers="host.docker.internal:9092",
    value_deserializer=lambda v: json.loads(v.decode("utf-8")),
    auto_offset_reset="earliest",
    enable_auto_commit=True,
)

print("Listening for predictions...\n")

for msg in consumer:
    print(msg.value)
