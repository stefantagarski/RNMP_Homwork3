import json
import time
import pandas as pd
from kafka import KafkaProducer

df = pd.read_csv("online.csv")

producer = KafkaProducer(
    bootstrap_servers="host.docker.internal:9092",
    value_serializer=lambda v: json.dumps(v).encode("utf-8")
)

for _, row in df.iterrows():
    record = row.drop("Diabetes_binary").to_dict()
    producer.send("health_data", record)
    time.sleep(1)
