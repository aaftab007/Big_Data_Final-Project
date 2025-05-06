# Streamlit App: Real-Time Fraud Detection with SparkML + Ngrok

import streamlit as st
import pandas as pd
import json
import time
import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StringType, DoubleType
from pyspark.sql.functions import col, when
from pyspark.ml import PipelineModel

# Set Streamlit title
st.title("🚨 Real-Time Fraud Detection App")

# Create stream_input folder if it doesn't exist
os.makedirs("stream_input", exist_ok=True)

# UI input form
with st.form("transaction_form"):
    st.subheader("📥 New Transaction")
    payment_type = st.selectbox("Payment Type", ["CREDIT", "DEBIT"])
    employment_status = st.selectbox("Employment Status", ["EMPLOYED", "UNEMPLOYED"])
    housing_status = st.selectbox("Housing Status", ["RENT", "OWN"])
    source = st.selectbox("Source", ["MOBILE", "WEB"])
    device_os = st.selectbox("Device OS", ["ANDROID", "IOS"])
    amount = st.number_input("Transaction Amount", min_value=0.01, value=100.0)
    submitted = st.form_submit_button("Submit Transaction")

if submitted:
    tx = {
        "payment_type": payment_type,
        "employment_status": employment_status,
        "housing_status": housing_status,
        "source": source,
        "device_os": device_os,
        "transaction_amount": amount
    }
    filename = f"stream_input/tx_{int(time.time())}.json"
    with open(filename, "w") as f:
        json.dump(tx, f)
    st.success(f"Transaction submitted to {filename}")

# Run Spark only once per session (or externally as a script)
if st.button("Start Streaming (console output)"):
    spark = SparkSession.builder.appName("RealTimeFraudDetection").getOrCreate()

    schema = StructType() \
        .add("payment_type", StringType()) \
        .add("employment_status", StringType()) \
        .add("housing_status", StringType()) \
        .add("source", StringType()) \
        .add("device_os", StringType()) \
        .add("transaction_amount", DoubleType())

    model = PipelineModel.load("sparkml_model")

    stream_df = spark.readStream.schema(schema).json("stream_input")

    predictions = model.transform(stream_df)
    labeled_df = predictions.withColumn("label", when(col("prediction") == 1.0, "FRAUD").otherwise("LEGIT"))
    count_df = labeled_df.groupBy("label").count()

    query = count_df.writeStream \
        .outputMode("complete") \
        .format("console") \
        .trigger(processingTime="5 seconds") \
        .start()

    query.awaitTermination()


