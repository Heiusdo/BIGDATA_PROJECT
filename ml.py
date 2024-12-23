import os
import zipfile
import findspark
findspark.init('C:\Spark\spark-3.5.3-bin-hadoop3\spark-3.5.3-bin-hadoop3')
import streamlit as st
from pyspark.sql import SparkSession, Row
from pyspark.ml.classification import LogisticRegressionModel
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.linalg import Vectors  # Add this import
os.environ["PYSPARK_PYTHON"] = "python"
# findspark.init('C:\Spark\spark-3.5.3-bin-hadoop3\spark-3.5.3-bin-hadoop3')
# Extract the model if not already extracted
def extract_model():
    if not os.path.exists('lr_model'):
        with zipfile.ZipFile('lr_model.zip', 'r') as zip_ref:
            zip_ref.extractall('lr_model')

# Initialize Spark session
def initialize_spark():
    return SparkSession.builder.appName("LogisticRegressionModel").getOrCreate()

# Load the model and data
def load_model_and_data(spark):
    # Load the model
    loaded_model = LogisticRegressionModel.load("lr_model")
    
    # Load the DataFrames
    train_data = spark.read.parquet("lr_model/train_data.parquet")
    test_data = spark.read.parquet("lr_model/test_data.parquet")
    
    return loaded_model, train_data, test_data

# Evaluate model performance
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

# Modified prediction function
def predict(spark, loaded_model, features):
    # Convert the list of integers to a DenseVector
    vector_features = Vectors.dense(features)
    # Create DataFrame with the proper vector format
    df = spark.createDataFrame([(vector_features,)], ["features"])
    predictions = loaded_model.transform(df)
    return predictions.select("prediction").collect()[0]["prediction"]

# Streamlit app
def main():
    st.set_page_config(page_title="ML Model Predictor", layout="wide")
    
    # Extract model and initialize Spark
    extract_model()
    spark = initialize_spark()
    
    # Load model and data
    loaded_model, train_data, test_data = load_model_and_data(spark)
    
    # Make predictions to evaluate
    predictions = loaded_model.transform(test_data)
    
    # Display model performance in a nice format
    st.title("Logistic Regression Model Performance")
    
    # Create columns for metrics
    col1, col2 = st.columns(2)
    
    performance_metrics = evaluate_model(predictions)
    with col1:
        st.metric(label="Accuracy", value=f"{performance_metrics['Accuracy']:.2%}")
        st.metric(label="Precision", value=f"{performance_metrics['Precision']:.2%}")
    with col2:
        st.metric(label="Recall", value=f"{performance_metrics['Recall']:.2%}")
        st.metric(label="Error Rate", value=f"{performance_metrics['Error Rate']:.2%}")
    
    # Add a separator
    st.markdown("---")
    
    # Prediction interface
    st.title("Make a Prediction")
    st.write("Enter integer values for each feature:")
    
    # Dynamically create input fields based on feature vector
    sample_features = predictions.select("features").first()["features"]
    feature_count = len(sample_features)
    
    # Create columns for feature inputs
    cols = st.columns(3)  # Arrange inputs in 3 columns
    features = []
    
    for i in range(feature_count):
        with cols[i % 3]:  # Distribute inputs across columns
            feature = st.number_input(
                f"Feature {i+1}",
                min_value=-100,  # Adjust these limits based on your needs
                max_value=100,
                value=0,
                step=1,
                format="%d",
                help=f"Enter an integer value for feature {i+1}"
            )
            features.append(int(feature))
    
    # Center the predict button
    col1, col2, col3 = st.columns([1,1,1])
    with col2:
        predict_button = st.button("Predict", use_container_width=True)
    
    # Show prediction result
    if predict_button:
        try:
            with st.spinner("Making prediction..."):
                prediction = predict(spark, loaded_model, features)
            st.success(f"Prediction Result: {int(prediction)}")
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.info("Please check your input values and try again.")

if __name__ == "__main__":
    main()
