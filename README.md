# Real-Time Bank Account Fraud Detection System


---

## Project Overview

This repository hosts an end-to-end, scalable real-time fraud detection pipeline for bank account transactions. We leverage Apache Spark Structured Streaming for ingesting live transaction data, a pre-trained XGBoost model for classification, and a Streamlit-based web dashboard for interactive monitoring and demo.

Key components:

* **train\_model.py**: SparkML pipeline to preprocess data and train a RandomForest model (for illustration). Generates a SparkML `PipelineModel`.
* **stream\_job.py**: Spark Structured Streaming job that reads incoming JSON/CSV transactions, applies feature engineering, loads the pickled XGBoost model (`xgb_adasyn_model.pkl`), and outputs fraud counts to the console.
* **real\_time\_app.py**: Streamlit application allowing users to submit individual transactions via a web form, write them to the streaming input folder, and trigger live Spark streaming from within the UI.
* **xgb\_adasyn\_model.pkl**: Pre-trained XGBoost classifier serialized with `pickle`.
* **sample\_input.csv**: Example batch of transactions to seed the streaming source for testing.

---

## Architecture

```text
+-------------+          +-----------------------+          +-------------+
| Transaction | --write->| Spark Structured     | --calls-> | XGBoost     |
| Generator   |          | Streaming + SparkML   |           | Model       |
| (JSON/CSV)  |          | (feature eng., load   |           | (.pkl)      |
+-------------+          |  pipeline)            |          +-------------+
                              |                                
                              v                                
                       +--------------+                         
                       | Console &    |                         
                       | Mongo Sink   |                         
                       +--------------+                         
                              ^                                
                              |                                
                      +-----------------+                      
                      | Streamlit UI    |                      
                      | (transaction    |                      
                      | form + logs)    |                      
                      +-----------------+                      
```

---

## Prerequisites

* **Google Cloud Project** with a Dataproc cluster (Spark 3.x, Python 3.x).
* **Local or cluster** pre-installed: Python 3.7+, Java 8+, Apache Spark, PySpark.
* **Python packages**: `xgboost`, `pyspark`, `streamlit`, `google-cloud-storage`, `pyngrok` (for demo).
* **GCS bucket** (optional) for model storage and streaming source.

---

## Setup & Installation

1. **Clone this repository**:

```bash
git clone https://github.com/your-org/fraud-detection-streaming.git
cd fraud-detection-streaming
```

2. **Upload the pickled XGBoost model** to your Dataproc master node or a GCS bucket:

```bash
# To local home directory on master:
scp xgb_adasyn_model.pkl <dataproc-master>:/home/$USER/
# Or to GCS:
gsutil cp xgb_adasyn_model.pkl gs://your-bucket/models/
```

3. **Prepare the streaming input folder**:

```bash
# On HDFS (recommended):
hdfs dfs -mkdir -p /user/$USER/bank_fraud/stream_input
# Or local FS:
mkdir -p ~/bank_fraud/stream_input
```

4. **Copy sample data**:

```bash
gsutil cp sample_input.csv gs://your-bucket/data/stream_input/  # if using GCS
# Or locally:
cp sample_input.csv ~/bank_fraud/stream_input/
```

---

## Training the Model (Optional)

> The pre-trained XGBoost model is provided; use this step only if you wish to retrain or experiment.

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  train_model.py
```

This will output a SparkML `PipelineModel` at `sparkml_model/`.

---

## Running the Streaming Job

Submit the Spark Structured Streaming application to process live transactions:

```bash
spark-submit \
  --master yarn \
  --deploy-mode client \
  --files xgb_adasyn_model.pkl \
  --packages \
    org.mongodb.spark:mongo-spark-connector_2.12:3.0.1,\
    ml.dmlc:xgboost4j-spark_2.12:1.6.2 \
  stream_job.py \
  --input hdfs:///user/$USER/bank_fraud/stream_input/ \
  --checkpoint hdfs:///user/$USER/bank_fraud/checkpoint/
```

Watch the console for real-time batch counts of `FRAUD` vs `LEGIT` transactions.

---

## Running the Dashboard

Launch the Streamlit app to submit test transactions and view live logs:

```bash
pip install -r requirements.txt
streamlit run real_time_app.py
```

* Navigate to the URL shown (default `http://localhost:8501`).
* Submit transactions via the form; each will appear in the live Spark stream.
* (Optional) Expose via ngrok:

  ```bash
  ngrok http 8501
  ```

---

## Sample Input Format

The streaming job ingests JSON records with the following schema:

```json
{
  "payment_type": "CREDIT",
  "employment_status": "EMPLOYED",
  "housing_status": "RENT",
  "source": "MOBILE",
  "device_os": "ANDROID",
  "transaction_amount": 100.0
}
```

A CSV alternative (`sample_input.csv`) with matching headers can also be dropped into the input folder:

```
payment_type,employment_status,housing_status,source,device_os,transaction_amount
CREDIT,EMPLOYED,RENT,MOBILE,ANDROID,100.0
DEBIT,UNEMPLOYED,OWN,WEB,IOS,300.0
```

---

## Results & Screenshots

* **Live Spark Console Output** showing batch counts and latency warnings.
* **Streamlit Dashboard** for real-time transaction submission and fraud counts.

![Console Output](images/console_output.png)
![Streamlit UI](images/streamlit_ui.png)

---

## Contributing

Contributions are welcome! Please open an issue or submit a pull request with improvements or fixes.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
