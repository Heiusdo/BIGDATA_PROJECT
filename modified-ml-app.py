import os
import zipfile
import findspark
findspark.init('C:\Spark\spark-3.5.3-bin-hadoop3\spark-3.5.3-bin-hadoop3')
import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors
os.environ["PYSPARK_PYTHON"] = "python"

# Feature definitions
FEATURE_COLUMNS = [
    "default", "housing", "loan", "customerID", "job_admin", "job_blue-collar",
    "job_entrepreneur", "job_housemaid", "job_management", "job_retired",
    "job_self-employed", "job_services", "job_student", "job_technician",
    "job_unemployed", "marital_divorced", "marital_married", "marital_single",
    "education_primary", "education_secondary", "education_tertiary",
    "education_unknown", "contact_cellular", "contact_telephone",
    "contact_unknown", "month_apr", "month_aug", "month_dec",
    "month_feb", "month_jan", "month_jul", "month_jun", "month_mar",
    "month_may", "month_nov", "month_oct", "month_sep",
    "poutcome_failure", "poutcome_other", "poutcome_success",
    "poutcome_unknown", "age", "balance", "day", "duration",
    "campaign", "pdays", "previous"
]

# Define which features are binary and which are numeric
BINARY_FEATURES = [
    "default", "housing", "loan", "job_admin", "job_blue-collar", "job_entrepreneur", "job_housemaid",
    "job_management", "job_retired", "job_self-employed", "job_services",
    "job_student", "job_technician", "job_unemployed",
    "marital_divorced", "marital_married", "marital_single",
    "education_primary", "education_secondary", "education_tertiary",
    "education_unknown", "contact_cellular", "contact_telephone",
    "contact_unknown", "month_apr", "month_aug", "month_dec",
    "month_feb", "month_jan", "month_jul", "month_jun", "month_mar",
    "month_may", "month_nov", "month_oct", "month_sep",
    "poutcome_failure", "poutcome_other", "poutcome_success",
    "poutcome_unknown"
]

NUMERIC_FEATURES = ["customerID","age", "balance", "day", "duration", "campaign", "pdays", "previous"]

def extract_model():
    if not os.path.exists('lr_model'):
        with zipfile.ZipFile('lr_model.zip', 'r') as zip_ref:
            zip_ref.extractall('lr_model')

def initialize_spark():
    return SparkSession.builder.appName("LogisticRegressionModel").getOrCreate()

def load_model_and_data(spark):
    loaded_model = LogisticRegressionModel.load("lr_model")
    train_data = spark.read.parquet("lr_model/train_data.parquet")
    test_data = spark.read.parquet("lr_model/test_data.parquet")
    return loaded_model, train_data, test_data

def evaluate_model(predictions):
    evaluator = MulticlassClassificationEvaluator(
        labelCol="label",
        predictionCol="prediction",
        metricName="accuracy"
    )
    
    accuracy = evaluator.evaluate(predictions)
    recall = evaluator.evaluate(predictions, {evaluator.metricName: "recallByLabel"})
    precision = evaluator.evaluate(predictions, {evaluator.metricName: "precisionByLabel"})
    error_rate = evaluator.evaluate(predictions, {evaluator.metricName: "weightedFalsePositiveRate"})
    
    return {
        "Accuracy": accuracy,
        "Recall": recall,
        "Precision": precision,
        "Error Rate": error_rate
    }

def predict(spark, loaded_model, features):
    vector_features = Vectors.dense(features)
    df = spark.createDataFrame([(vector_features,)], ["features"])
    predictions = loaded_model.transform(df)
    return predictions.select("prediction").collect()[0]["prediction"]

def main():
    st.set_page_config(page_title="Bank Marketing ML Model Predictor", layout="wide")
    
    extract_model()
    spark = initialize_spark()
    loaded_model, train_data, test_data = load_model_and_data(spark)
    predictions = loaded_model.transform(test_data)
    
    st.title("Bank Marketing Prediction Model")
    st.markdown("This model predicts whether a client will subscribe to a term deposit.")
    
    # Model performance metrics
    st.header("Model Performance Metrics")
    col1, col2 = st.columns(2)
    performance_metrics = evaluate_model(predictions)
    with col1:
        st.metric(label="Accuracy", value=f"{performance_metrics['Accuracy']:.2%}")
        st.metric(label="Precision", value=f"{performance_metrics['Precision']:.2%}")
    with col2:
        st.metric(label="Recall", value=f"{performance_metrics['Recall']:.2%}")
        st.metric(label="Error Rate", value=f"{performance_metrics['Error Rate']:.2%}")
    
    st.markdown("---")
    
    # Prediction interface
    st.header("Make a Prediction")
    st.markdown("Enter values for each feature to get a prediction.")
    
    # Create tabs for different feature categories
    binary_tab, numeric_tab = st.tabs(["Categorical Features (0/1)", "Numeric Features"])
    
    features_dict = {}
    
    # Binary features input
    with binary_tab:
        st.markdown("Select 0 (No) or 1 (Yes) for each categorical feature:")
        cols = st.columns(3)
        for idx, feature in enumerate(BINARY_FEATURES):
            with cols[idx % 3]:
                features_dict[feature] = st.selectbox(
                    feature,
                    options=[0, 1],
                    help=f"Select 0 for No or 1 for Yes"
                )
    
    # Numeric features input
    with numeric_tab:
        st.markdown("Enter values for numeric features:")
        cols = st.columns(3)
        for idx, feature in enumerate(NUMERIC_FEATURES):
            with cols[idx % 3]:
                features_dict[feature] = st.number_input(
                    feature,
                    min_value=-100000,
                    max_value=100000,
                    value=0,
                    help=f"Enter a numeric value for {feature}"
                )
    
    # Prepare features in the correct order
    features = [features_dict[col] for col in FEATURE_COLUMNS]
    
    # Center the predict button
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        predict_button = st.button("Predict", use_container_width=True)
    
    if predict_button:
        try:
            with st.spinner("Making prediction..."):
                prediction = predict(spark, loaded_model, features)
            result = "Yes" if prediction == 1 else "No"
            if result == "Yes":
                st.success("Prediction: Client will subscribe to the term deposit")
            else:
                st.success(f"Prediction: Client wont subscribe to the term deposit")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()
