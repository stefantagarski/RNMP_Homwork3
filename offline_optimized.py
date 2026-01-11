from pyspark.sql import SparkSession
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier, DecisionTreeClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator, BinaryClassificationEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.tuning import ParamGridBuilder, CrossValidator
import time


def create_spark_session():
    """Create optimized Spark session with performance tuning"""
    return SparkSession.builder \
        .appName("DiabetesOfflineOptimized") \
        .config("spark.sql.shuffle.partitions", "8") \
        .config("spark.sql.adaptive.enabled", "true") \
        .config("spark.sql.adaptive.coalescePartitions.enabled", "true") \
        .config("spark.serializer", "org.apache.spark.serializer.KryoSerializer") \
        .config("spark.kryoserializer.buffer.max", "512m") \
        .getOrCreate()


def load_and_prepare_data(spark, file_path):
    """Load data and perform basic validation"""
    df = spark.read.csv(file_path, header=True, inferSchema=True)
    
    print(f"Dataset loaded: {df.count()} rows, {len(df.columns)} columns")
    print("\nClass distribution:")
    df.groupBy("Diabetes_binary").count().show()
    
    # Cache the data for better performance
    df = df.cache()
    
    return df


def create_preprocessing_stages(feature_cols):
    """Create reusable preprocessing pipeline stages"""
    assembler = VectorAssembler(
        inputCols=feature_cols,
        outputCol="features_vec",
        handleInvalid="skip"  # Handle invalid data gracefully
    )
    
    scaler = StandardScaler(
        inputCol="features_vec",
        outputCol="features",
        withMean=True,
        withStd=True
    )
    
    return assembler, scaler


def get_model_configs(assembler, scaler):
    """Define model configurations with extended hyperparameter grids"""
    
    # Logistic Regression
    lr = LogisticRegression(featuresCol="features", labelCol="Diabetes_binary", maxIter=100)
    lr_pipeline = Pipeline(stages=[assembler, scaler, lr])
    lr_params = ParamGridBuilder() \
        .addGrid(lr.regParam, [0.001, 0.01, 0.1]) \
        .addGrid(lr.elasticNetParam, [0.0, 0.5, 1.0]) \
        .build()
    
    # Random Forest
    rf = RandomForestClassifier(featuresCol="features", labelCol="Diabetes_binary", seed=42)
    rf_pipeline = Pipeline(stages=[assembler, scaler, rf])
    rf_params = ParamGridBuilder() \
        .addGrid(rf.numTrees, [20, 50, 100]) \
        .addGrid(rf.maxDepth, [5, 10, 15]) \
        .addGrid(rf.minInstancesPerNode, [1, 5]) \
        .build()
    
    # Decision Tree
    dt = DecisionTreeClassifier(featuresCol="features", labelCol="Diabetes_binary", seed=42)
    dt_pipeline = Pipeline(stages=[assembler, scaler, dt])
    dt_params = ParamGridBuilder() \
        .addGrid(dt.maxDepth, [5, 10, 15, 20]) \
        .addGrid(dt.minInstancesPerNode, [1, 5, 10]) \
        .addGrid(dt.maxBins, [32, 64]) \
        .build()
    
    return [
        ("Logistic Regression", lr_pipeline, lr_params),
        ("Random Forest", rf_pipeline, rf_params),
        ("Decision Tree", dt_pipeline, dt_params)
    ]


def evaluate_model(model, data, label_col):
    """Comprehensive model evaluation with multiple metrics"""
    predictions = model.transform(data)
    
    # F1 Score
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        metricName="f1"
    )
    f1 = f1_evaluator.evaluate(predictions)
    
    # Accuracy
    acc_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        metricName="accuracy"
    )
    accuracy = acc_evaluator.evaluate(predictions)
    
    # Precision
    prec_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        metricName="weightedPrecision"
    )
    precision = prec_evaluator.evaluate(predictions)
    
    # Recall
    rec_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        metricName="weightedRecall"
    )
    recall = rec_evaluator.evaluate(predictions)
    
    # AUC
    auc_evaluator = BinaryClassificationEvaluator(
        labelCol=label_col,
        metricName="areaUnderROC"
    )
    auc = auc_evaluator.evaluate(predictions)
    
    return {
        "f1": f1,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "auc": auc
    }


def train_and_evaluate_models(df, model_configs, label_col):
    """Train models with cross-validation and return the best one"""
    
    f1_evaluator = MulticlassClassificationEvaluator(
        labelCol=label_col,
        metricName="f1"
    )
    
    best_model = None
    best_metrics = None
    best_name = ""
    best_f1 = 0.0
    
    results = []
    
    for name, pipeline, param_grid in model_configs:
        print(f"\n{'='*60}")
        print(f"Training: {name}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Create cross-validator
        cv = CrossValidator(
            estimator=pipeline,
            estimatorParamMaps=param_grid,
            evaluator=f1_evaluator,
            numFolds=5,
            parallelism=2,  # Parallel fold execution
            seed=42
        )
        
        # Train model
        cv_model = cv.fit(df)
        
        training_time = time.time() - start_time
        
        # Evaluate
        metrics = evaluate_model(cv_model.bestModel, df, label_col)
        
        print(f"\nResults for {name}:")
        print(f"  F1 Score:  {metrics['f1']:.4f}")
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  AUC-ROC:   {metrics['auc']:.4f}")
        print(f"  Training time: {training_time:.2f}s")
        
        results.append({
            "name": name,
            "metrics": metrics,
            "training_time": training_time
        })
        
        # Track best model
        if metrics['f1'] > best_f1:
            best_f1 = metrics['f1']
            best_model = cv_model.bestModel
            best_metrics = metrics
            best_name = name
    
    return best_model, best_name, best_metrics, results


def save_model(model, model_name, base_path="saved_models"):
    """Save the trained model"""
    model_path = f"{base_path}/best_{model_name.lower().replace(' ', '_')}"
    model.write().overwrite().save(model_path)
    print(f"\nModel saved to: {model_path}")
    return model_path


def print_final_summary(best_name, best_metrics, all_results):
    """Print comprehensive summary of all models"""
    print("\n" + "="*60)
    print("FINAL SUMMARY")
    print("="*60)
    
    print("\nAll Models Performance:")
    print("-" * 60)
    for result in all_results:
        print(f"\n{result['name']}:")
        print(f"  F1: {result['metrics']['f1']:.4f} | "
              f"Acc: {result['metrics']['accuracy']:.4f} | "
              f"AUC: {result['metrics']['auc']:.4f}")
    
    print("\n" + "="*60)
    print(f"BEST MODEL: {best_name}")
    print("="*60)
    print(f"F1 Score:  {best_metrics['f1']:.4f}")
    print(f"Accuracy:  {best_metrics['accuracy']:.4f}")
    print(f"Precision: {best_metrics['precision']:.4f}")
    print(f"Recall:    {best_metrics['recall']:.4f}")
    print(f"AUC-ROC:   {best_metrics['auc']:.4f}")
    print("="*60)


def main():
    """Main execution function"""
    
    # Create Spark session
    spark = create_spark_session()
    
    try:
        # Load and prepare data
        df = load_and_prepare_data(spark, "offline.csv")
        
        # Define features
        label_col = "Diabetes_binary"
        feature_cols = [c for c in df.columns if c != label_col]
        
        print(f"\nFeatures ({len(feature_cols)}): {', '.join(feature_cols)}")
        
        # Create preprocessing stages
        assembler, scaler = create_preprocessing_stages(feature_cols)
        
        model_configs = get_model_configs(assembler, scaler)
        
        best_model, best_name, best_metrics, all_results = train_and_evaluate_models(
            df, model_configs, label_col
        )
        
        save_model(best_model, best_name)
        
        print_final_summary(best_name, best_metrics, all_results)
        
    finally:
        spark.stop()


if __name__ == "__main__":
    main()
