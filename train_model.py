# Train SparkML Model for Fraud Detection

from pyspark.sql import SparkSession
from pyspark.ml import Pipeline
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.feature import StringIndexer, OneHotEncoder, VectorAssembler
from pyspark.sql.types import *

# Start Spark
spark = SparkSession.builder.appName("TrainModel").getOrCreate()

# Sample Data
data = [
    ("CREDIT", "EMPLOYED", "RENT", "MOBILE", "ANDROID", 100.0, 0),
    ("DEBIT", "UNEMPLOYED", "OWN", "WEB", "IOS", 300.0, 1),
]
schema = StructType([
    StructField("payment_type", StringType()),
    StructField("employment_status", StringType()),
    StructField("housing_status", StringType()),
    StructField("source", StringType()),
    StructField("device_os", StringType()),
    StructField("transaction_amount", DoubleType()),
    StructField("fraud_bool", IntegerType()),
])
df = spark.createDataFrame(data, schema)

# Preprocessing
cat_cols = ['payment_type', 'employment_status', 'housing_status', 'source', 'device_os']
indexers = [StringIndexer(inputCol=col, outputCol=f"{col}_idx") for col in cat_cols]
encoders = [OneHotEncoder(inputCol=f"{col}_idx", outputCol=f"{col}_vec") for col in cat_cols]
assembler = VectorAssembler(inputCols=[f"{col}_vec" for col in cat_cols] + ['transaction_amount'], outputCol="features")

rf = RandomForestClassifier(labelCol="fraud_bool", featuresCol="features")
pipeline = Pipeline(stages=indexers + encoders + [assembler, rf])

model = pipeline.fit(df)
model.save("sparkml_model")
print("✅ Model saved to 'sparkml_model'")
