from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col
from pyspark.sql.types import StructType, DoubleType
from pyspark.ml import PipelineModel

spark = SparkSession.builder.appName("OnlinePrediction").getOrCreate()

schema = StructType() \
    .add("HighBP", DoubleType()) \
    .add("HighChol", DoubleType()) \
    .add("CholCheck", DoubleType()) \
    .add("BMI", DoubleType()) \
    .add("Smoker", DoubleType()) \
    .add("Stroke", DoubleType()) \
    .add("HeartDiseaseorAttack", DoubleType()) \
    .add("PhysActivity", DoubleType()) \
    .add("Fruits", DoubleType()) \
    .add("Veggies", DoubleType()) \
    .add("HvyAlcoholConsump", DoubleType()) \
    .add("AnyHealthcare", DoubleType()) \
    .add("NoDocbcCost", DoubleType()) \
    .add("GenHlth", DoubleType()) \
    .add("MentHlth", DoubleType()) \
    .add("PhysHlth", DoubleType()) \
    .add("DiffWalk", DoubleType()) \
    .add("Sex", DoubleType()) \
    .add("Age", DoubleType()) \
    .add("Education", DoubleType()) \
    .add("Income", DoubleType())

model = PipelineModel.load("saved_models/best_dt")

stream_df = spark.readStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "host.docker.internal:9092") \
    .option("subscribe", "health_data") \
    .load()

parsed = stream_df.select(
    from_json(col("value").cast("string"), schema).alias("data")
).select("data.*")

predicted = model.transform(parsed)

output = predicted.selectExpr("to_json(struct(*)) AS value")

query = output.writeStream \
    .format("kafka") \
    .option("kafka.bootstrap.servers", "host.docker.internal:9092") \
    .option("topic", "health_data_predicted") \
    .option("checkpointLocation", "/tmp/checkpoint") \
    .start()

query.awaitTermination()
