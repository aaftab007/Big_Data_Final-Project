
from pyspark.sql import SparkSession
from pyspark.sql.functions import from_json, col, udf
from pyspark.sql.types import StructType, DoubleType, IntegerType
import joblib
import numpy as np

# Load model using joblib
model = joblib.load("fraud_model.pkl")

# Define prediction UDF
def predict_fraud_udf(*cols):
    features = np.array(cols).reshape(1, -1)
    pred = model.predict(features)
    return int(pred[0])

# Register as UDF
predict_udf = udf(predict_fraud_udf, IntegerType())

# Define your Spark Session
spark = SparkSession.builder     .appName("FraudDetectionStreaming")     .config("spark.mongodb.output.uri", "mongodb://your_mongo_host:27017/fraud.results")     .getOrCreate()

spark.sparkContext.setLogLevel("WARN")

# Define Kafka source
df_raw = spark.readStream     .format("kafka")     .option("kafka.bootstrap.servers", "localhost:9092")     .option("subscribe", "fraud_topic")     .load()

# Define schema matching incoming stream
schema = StructType() \
    .add("feature1", DoubleType()) \
    .add("feature2", DoubleType()) \
    .add("feature3", DoubleType()) \
    .add("feature4", DoubleType())

# Parse the Kafka value (assumes JSON string)
df_parsed = df_raw.selectExpr("CAST(value AS STRING)") \
    .select(from_json("value", schema).alias("data")) \
    .select("data.*")

# Apply model prediction
df_scored = df_parsed.withColumn("prediction", predict_udf(*[col(c) for c in df_parsed.columns]))

# Write to MongoDB
query = df_scored.writeStream \
    .format("mongodb") \
    .option("checkpointLocation", "/tmp/fraud_checkpoint") \
    .outputMode("append") \
    .start()

query.awaitTermination()
