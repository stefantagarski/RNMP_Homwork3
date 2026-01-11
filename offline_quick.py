"""
Quick Offline Training - Minimal hyperparameters (2-3 minutes)
"""

from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml import Pipeline
import time

print("Starting quick training...")

# Create Spark session
spark = SparkSession.builder \
    .appName("DiabetesQuickTraining") \
    .config("spark.sql.shuffle.partitions", "4") \
    .getOrCreate()

# Load data
df = spark.read.csv("offline.csv", header=True, inferSchema=True).cache()
print(f"✓ Dataset loaded: {df.count()} rows")

# Split data
train_df, test_df = df.randomSplit([0.8, 0.2], seed=42)
print(f"✓ Train: {train_df.count()} | Test: {test_df.count()}")

# Define features
label_col = "Diabetes_binary"
feature_cols = [c for c in df.columns if c != label_col]

# Create pipeline
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features_vec")
scaler = StandardScaler(inputCol="features_vec", outputCol="features", withMean=True, withStd=True)
dt = DecisionTreeClassifier(featuresCol="features", labelCol=label_col, maxDepth=15, seed=42)

pipeline = Pipeline(stages=[assembler, scaler, dt])

# Train
print("\n" + "="*60)
print("Training Decision Tree...")
print("="*60)

start = time.time()
model = pipeline.fit(train_df)
training_time = time.time() - start

print(f"✓ Training completed in {training_time:.2f}s")

# Evaluate
predictions = model.transform(test_df)
evaluator = MulticlassClassificationEvaluator(labelCol=label_col, metricName="f1")
f1 = evaluator.evaluate(predictions)

acc_eval = MulticlassClassificationEvaluator(labelCol=label_col, metricName="accuracy")
accuracy = acc_eval.evaluate(predictions)

print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"F1 Score:  {f1:.4f}")
print(f"Accuracy:  {accuracy:.4f}")
print("="*60)

# Save model
model_path = "saved_models/best_decision_tree"
model.write().overwrite().save(model_path)
print(f"\n✓ Model saved to: {model_path}")

spark.stop()
print("\n✓ Done!")
