import streamlit as st
import pandas as pd

import streamlit as st

# Define custom CSS to hide the Streamlit header and footer
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
"""

# Inject custom CSS to hide Streamlit UI elements
st.markdown(hide_streamlit_style, unsafe_allow_html=True)

# Custom HTML for header and footer
footer_html = """
    <div style="position: fixed; bottom: 0; width: 50%; background-color: black; z-index: 9999; border-bottom: 1px solid #ddd;">
        <div style="max-width: 100%; margin: 0 auto; padding: 10px 0;">
            <h5 style="text-align: center; color: #FFD700; margin: 0; padding: 1px 0;">VIETNAM NATIONAL UNIVERSITY OF HO CHI MINH CITY - INTERNATIONAL UNIVERSITY</h4>
            <h5 style="text-align: center; color: #FFD700; margin: 0; padding: 1px 0;">SCHOOL OF COMPUTER SCIENCE AND ENGINEERING</h4>
        </div>
    </div>
    <br><br><br><br><br>
"""

# Custom HTML for the title
title_html = """
    <h1 style="color: green; text-align: center;">
        Bank Marketing Dataset Report
    </h1>
"""

# Display the custom HTML for the title
st.markdown(title_html, unsafe_allow_html=True)

st.write("Welcome to the Bank Marketing Dataset Report. This project delves into a comprehensive analysis of a bank marketing dataset, exploring various attributes and behaviors of customers. By examining key variables such as age, financial balances, call durations, and campaign contacts, we aim to uncover critical insights and predictors of customer outcomes. Through meticulous data preprocessing, exploratory data analysis, and advanced visualization techniques, this report offers a deep understanding of the factors influencing customer behavior and the effectiveness of marketing strategies. Join us as we navigate through the data to reveal patterns and trends that drive successful marketing campaigns.")

import streamlit as st

# Create a 4 columns by 3 rows table
table_html = """
    <table style="width:100%; border: 1px solid black; border-collapse: collapse;">
        <tr>
            <th style="border: 1px solid black;">Name</th>
            <th style="border: 1px solid black;">ID</th>
            <th style="border: 1px solid black;">Contribution</th>
            <th style="border: 1px solid black;">Note</th>
        </tr>
        <tr>
            <td style="border: 1px solid black;">Hoang Van Long</td>
            <td style="border: 1px solid black;">ITDSIU21096</td>
            <td style="border: 1px solid black;">Organizing data, plans, code lab report, preprocessing</td>
            <td style="border: 1px solid black;"></td>
        </tr>
        <tr>
            <td style="border: 1px solid black;">Do Minh Hieu</td>
            <td style="border: 1px solid black;">ITDSIU21086</td>
            <td style="border: 1px solid black;">Training model</td>
            <td style="border: 1px solid black;"></td>
        </tr>
        <tr>
            <td style="border: 1px solid black;">Nguyen Mai Anh Nam</td>
            <td style="border: 1px solid black;">ITDSIU21102</td>
            <td style="border: 1px solid black;">Building application</td>
            <td style="border: 1px solid black;"></td>
        </tr>
    </table>
"""

# Display the table
st.markdown(table_html, unsafe_allow_html=True)

# Read the main csv file
df = pd.read_csv('bank.csv')

# Define the sections
sections = {
    "I. Introduction": "#introduction",
    "II. Data Collecting and Manipulation": "#section1",
    "III. Exploratory Data Analysis (EDA)": "#section2",
    "IV. Data Preprocessing": "#section3",
    "V. Model Development": "#section4",
    "VI. Application": "#section5",
    "VII. Conclusion": "#section6",
    "VIII. References": "#references"
}

# Create the Table of Contents at the top
st.title("Table of Contents")
for section, link in sections.items():
    st.markdown(f"[{section}]({link})")
    
# Function to create styled header
def styled_header(text):
    html_header = f"""
        <h2 style="color: gold;">{text}</h2>
    """
    st.markdown(html_header, unsafe_allow_html=True)


# I. Introduction
introduction = {
    "Overview": {
        "Dataset": "[**Bank Marketing Dataset**](https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data)",
        "Dataset Overview": "This dataset contains data on a bank's Term Deposit ([Term Deposit](https://timo.vn/tai-khoan-tiet-kiem/khi-nao-nen-su-dung-goal-save-va-term-deposit/)) product marketing strategy.",
        "Source": "[Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014"
    },
    "Problem and Solution": {
        "Classification Problem": "Predict whether a customer will decide to enroll in this program '**Deposit**'/'**No Deposit**' (2-class classification problem) through the characteristics of that customer.",
        "Overview of the data set analysis plan in this report": [
            "Data Collection",
            "Data Manipulation: feature extraction, normalization",
            "Complementation: 'Null/None/NA' handling",
            "**Statistics:**",
            "Statistics on data/label volume",
            "Feature distribution, correlation, etc",
            "Distribution charts",
            "**Cleaning:** Data pre-processing: normalization, splitting the training test set, dropping irrelevant features, etc",
            "**Feature extraction:** Encoding features into feature vectors and label vectors for computation",
            "**Analysis/forecasting using machine learning algorithms:**",
            "Selecting some techniques",
            "Evaluating effectiveness",
            "Using metrics appropriate to the problem"
        ]
    },
    "Dataset Description": {
        "Description of information of some fields in the data set": [
            "* **age**: Represents the age of the customer.",
            "* **job**: Describes the person's occupation.",
            "* **marital**: Indicates the marital status of the person (e.g., married, single, divorced).",
            "* **education**: Represents the person's level of education (e.g., primary, secondary, tertiary).",
            "* **default**: Indicates whether the person has a credit card ('yes', 'no', or 'unknown').",
            "* **housing**: Indicates whether the person has a housing loan ('yes', 'no', or 'unknown').",
            "* **loan**: Indicates whether the person has a personal loan ('yes', 'no', or 'unknown').",
            "* **contact**: Describes the communication method used to contact the person (e.g., 'cellular', 'telephone').",
            "* **day**: Indicates the day of the week of the last contact.",
            "* **month**: Indicates the month of the last contact.",
            "* **duration**: Duration of the last contact in seconds.",
            "* **campaign**: The number of contacts performed during this campaign.",
            "* **pdays**: Represents the number of days since the last contact with the person, or -1 if they were never contacted before.",
            "* **previous**: The number of contacts performed before this campaign.",
            "* **poutcome**: Describes the outcome of the previous marketing campaign.",
            "* **deposit**: The target variable that indicates whether the person subscribed to a savings account ('yes' or 'no')."
        ]
    }
}

# Set up the UI
st.markdown("<a name='introduction'></a>", unsafe_allow_html=True)
styled_header("I. Introduction")
# Display Overview section
st.subheader("1.1. Overview")
st.write("**Dataset:**", introduction["Overview"]["Dataset"])
st.write("**Dataset Overview:**", introduction["Overview"]["Dataset Overview"])
st.write("**Source:**", introduction["Overview"]["Source"])

# Display Problem and Solution section
st.subheader("1.2. Problem and Solution")
st.write("**Classification Problem:**", introduction["Problem and Solution"]["Classification Problem"])
st.write("**Overview of the data set analysis plan in this report:**")
for item in introduction["Problem and Solution"]["Overview of the data set analysis plan in this report"]:
    if "**" in item:
        st.markdown(f"{item}")
    else:
        st.markdown(f"* {item}")

# Display Dataset Description section
st.subheader("1.3. Dataset Description")
st.write("**Description of information of some fields in the data set:**")
for item in introduction["Dataset Description"]["Description of information of some fields in the data set"]:
    st.markdown(f"{item}")
    

# Read the text file
with open('pic/overall.txt', 'r', encoding='utf-8') as file:
    overall_text = file.read()
with open('pic/checkingNull.txt', 'r', encoding='utf-8') as file:
    checkingNull_text = file.read()
with open('pic/jobCount.txt', 'r', encoding='utf-8') as file:
    jobCount_text = file.read()

    
# II. Data Collecting and Manipulation
dataCollectionAndManipulation = {
    "Read Data": {
        "Dataset": df.head(10),  # Display the first 10 rows of the dataframe
        "Overall Dataset": overall_text,  # Contents of the overall.txt file
        "Check null and duplicated variables": checkingNull_text # Contents of the checkingNull.txt file
    },
    "Data Manipulation": {
        "Count value of 'job' attribute": jobCount_text, # Contents of the jobCount.txt file
        "Remove unknown values and special characters from the 'job' column": "Python code",
        "Separating": {
            "Deposit and non-deposit customers": "Python code",
            "Categorical and numerical columns": "Python code"
        }
    }
}

# Displaying the UI for section II
st.markdown("<a name='section1'></a>", unsafe_allow_html=True)
styled_header("II. Data Collecting and Manipulation")
# Display Read Data section
st.subheader("2.1. Read Data")
st.write("**Dataset:**")
st.dataframe(dataCollectionAndManipulation["Read Data"]["Dataset"])

st.write("**Overall Dataset:**")
st.markdown(f"```{dataCollectionAndManipulation["Read Data"]["Overall Dataset"]}```")

st.write("**Check null and duplicated variables:**")
st.markdown(f"```{dataCollectionAndManipulation["Read Data"]["Check null and duplicated variables"]}```")

# Display Data Manipulation section
st.subheader("2.2. Data Manipulation")
st.write("**Count value of 'job' attribute:**")
st.markdown(f"```{dataCollectionAndManipulation["Data Manipulation"]["Count value of \'job\' attribute"]}```")

st.write("**Remove unknown values and special characters from the 'job' column:**")
st.code("""df = df[df['job'] != 'unknown'] 
df.job = df.job.str.replace('.', '')""", language='python')

st.write("**Separating:**")
st.write("* Deposit and non-deposit customers")
st.code("""deposit = df[df["deposit"] == "yes"]
not_deposit = df[df["deposit"] == "no"]""", language='python')

st.write("* Categorical and numerical columns")
st.code("""Id_col = ['']
target_col = ["deposit"]
cat_cols = df.nunique()[df.nunique() < 6].keys().tolist()
cat_cols = [x for x in cat_cols if x not in target_col]
num_cols = [x for x in df.columns if (x not in cat_cols + target_col + Id_col)]""", language='python')

# III. Data Analysis
exploratoryDataAnalysis = {
    "Assess the level of \"customer attrition\" in the dataset": {
        "What is \'customer attrition\'?": "pic\deposit&nonDeposit.png",
        "Definition" : "=>Customer attrition is defined as the loss of customers by a business for whatever reason."
    },
    "Variables distribution in customer attrition" : {
        "Marital": "pic\martial_pie.png",
        "Education": "pic\education_pie.png",
        "Default": "pic\default_pie.png",
        "Housing": "pic\housing_pie.png",
        "Loan": "pic\loan_pie.png",
        "Contact": "pic\contact_pie.png",
        "Poutcome": "pic\poutcome_pie.png",
        "Age": r"pic\age_hist.png",
        "Job" : "pic\job_hist.png",
        "Balance": r"pic\balance_hist.png",
        "Day": "pic\day_hist.png",
        "Month": "pic\month_hist.png",
        "Duration": "pic\duration_hist.png",
        "Campaign": "pic\campaign_hist.png",
        "pday": "pic\pdays_hist.png",
        "Previous": "pic\previous_hist.png",
        "Scatter plot matrix for Numerical columns": r"pic\scatterMatrix.png",
        "Conclusion" : "=>  The scatter matrix shows that duration has the strongest correlation with the outcome, while variables like campaign and p-days exhibit no clear relationships."
    }
    
}

# Displaying the UI for section III
st.markdown("<a name='section2'></a>", unsafe_allow_html=True)
styled_header("III. Exploratory Data Analysis (EDA)")
st.subheader("3.1. Assess the level of \"customer attrition\" in the dataset:")
st.write("**What is \'customer attrition\'?**")
st.image(exploratoryDataAnalysis["Assess the level of \"customer attrition\" in the dataset"]["What is 'customer attrition'?"])
st.write(exploratoryDataAnalysis["Assess the level of \"customer attrition\" in the dataset"]["Definition"])

st.write("**Variables distribution in customer attrition:**")
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Marital"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Education"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Default"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Housing"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Loan"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Contact"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Poutcome"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Age"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Job"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Balance"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Day"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Month"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Duration"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Campaign"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["pday"])
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Previous"])
st.write("Scatter plot matrix for Numerical columns:")
st.image(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Scatter plot matrix for Numerical columns"])
st.write(exploratoryDataAnalysis["Variables distribution in customer attrition"]["Conclusion"])

# IV.	Data Preprocessing:
# Displaying the UI for section IV
st.markdown("<a name='section3'></a>", unsafe_allow_html=True)
styled_header("IV. Data Preprocessing")
st.write("**- In this step, I used scikit-learn to process data:**")
st.write("- Customer ID and target column")
st.code("""#customer id col
Id_col = ['customerID']
df['customerID'] = df.index

#Target columns
target_col = ["deposit"]""", language="python")
st.write("- Process Categorical and Numerical columns")
st.code("""#numerical columns
num_cols = [x for x in df.columns if x not in cat_cols + target_col + Id_col]

#Binary columns with 2 values
bin_cols = df.nunique()[df.nunique() == 2].keys().tolist()

#Columns more than 2 values
multi_cols = [i for i in cat_cols if i not in bin_cols]

#Label encoding Binary columns
le = LabelEncoder()
for i in bin_cols :
    df[i] = le.fit_transform(df[i])
    
#Duplicating columns for multi value columns
df = pd.get_dummies(data = df,columns = multi_cols )

#Scaling Numerical columns
std = StandardScaler()
scaled = std.fit_transform(df[num_cols])
scaled = pd.DataFrame(scaled,columns=num_cols)

#dropping original values merging scaled values for numerical columns
df_og = df.copy()
df = df.drop(columns = num_cols,axis = 1)
df = df.merge(scaled,left_index=True,right_index=True,how = "left")
df = df.dropna()""", language="python")

# 4.1
st.write("### 4.1. Variable Summary:")

st.image(r"pic\variable_summary.png")
st.write("The dataset consists of customer attributes and behaviors, showing notable variability in financial balances, call durations, and previous campaign contacts. Key observations include:")

st.write("- **Age**: Skewed toward younger adults, with a mean of 41.")
st.write("- **Balances**: High variability, ranging from significant debt to large positive balances.")
st.write("- **Call duration (duration)**: Strong indicator with a wide range (up to 3881 seconds).")
st.write("- **Campaign contacts**: Most customers were contacted very few times.")
st.write("- **Outcome (deposit)**: Balanced between success and failure (~47% subscribed).")

st.write("⇒ This suggests that variables like duration, balance, and previous contacts may be critical predictors of the target outcome.")

# 4.2
st.write("### 4.2. Correlation Matrix:")
st.image(r"pic\corr_matrix.png")
st.write("The correlation matrix highlights the relationships between variables:")

st.write("- **Strongest Correlations**: Duration shows a significant positive correlation with the outcome variables (poutcome_success and deposit), confirming its importance as a predictor.")
st.write("- **Weak Correlations**: Most variables, such as age, campaign, and pdays, exhibit low correlations with the outcome or with other features, indicating weak linear relationships.")
st.write("- **Multicollinearity**: Some categorical variables (e.g., marital, education) have moderate correlations within their categories, suggesting overlapping information.")
st.write("- **Binary Features**: Variables like loan, default, and housing show almost no correlation with the target variable (deposit), implying limited predictive power.")

st.write("⇒ In summary, duration remains the most influential variable, while most other features demonstrate low direct correlations with the target outcome.")

# 4.3
st.write("### 4.3. Visualizing Data with Principal Components:")
st.image(r"pic\visualization.png")
st.write("**=> We can see that the first principal component (PC1) is a strong discriminator between the two categories. It appears that higher values of PC1 are associated with \"not deposit\" cases, while lower values are associated with \"deposit\" cases.**")

# V. Logistic Regression Model
st.markdown("<a name='section4'></a>", unsafe_allow_html=True)
styled_header("V. Model Development")

# 5.1 Initializing Spark Session
st.subheader("5.1 Initializing Spark Session")
st.code("""
from pyspark.sql import SparkSession
spark = SparkSession.builder.appName("LogisticRegressionModel").getOrCreate()
""", language="python")
st.write("Creates Spark session to manage cluster resources and distributed processing.")

# 5.2 Reading CSV File Using Spark
st.subheader("5.2 Reading CSV File Using Spark")
st.code("""
df = spark.read.csv("newdata.csv", header=True, inferSchema=True)
""", language="python")
st.write("Reads the CSV file into a Spark DataFrame with headers and schema inference.")

# 5.3 Splitting Data
st.subheader("5.3 Splitting Data")
st.code("""
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression

# Prepare feature vector
assembler = VectorAssembler(inputCols=df.columns[:-1], outputCol="features")
df_features = assembler.transform(df).select("features", "label")

# Split the data
train_data, test_data = df_features.randomSplit([0.8, 0.2], seed=1234)
""", language="python")
st.write("Prepares feature vectors and splits the data into training and test sets.")

# Splitting Data Statistics
st.write("##### Splitting Data Statistic")
import streamlit as st

# Define the HTML with inline CSS for light blue color
training_html = """
    <div>Training Dataset Count: <span style="color: red;">8833</span></div>
"""
test_html = """
    <div>Test Dataset Count: <span style="color: red;">2189</span></div>
"""

# Display the HTML content
st.markdown(training_html, unsafe_allow_html=True)
st.markdown(test_html, unsafe_allow_html=True)


# 5.4 Training Model
st.subheader("5.4 Training Model")
st.code("""
lr = LogisticRegression(featuresCol='features', labelCol='label')
lr_model = lr.fit(train_data)
""", language="python")
st.write("Training a logistic regression model using the training data.")

# 5.5 Saving Model
st.subheader("5.5 Saving Model")
st.code("""
# Save the trained model
lr_model.save("lr_model")

# Save train and test datasets
train_data.write.parquet("lr_model/train_data.parquet")
test_data.write.parquet("lr_model/test_data.parquet")
""", language="python")
st.write("Saving the trained model and training/test datasets for future use.")

# 5.6 Prediction
st.subheader("5.6 Prediction")
st.code("""
# Load the saved model
from pyspark.ml.classification import LogisticRegressionModel
loaded_model = LogisticRegressionModel.load("lr_model")

# Perform predictions on test data
predictions = loaded_model.transform(test_data)
predictions.select("features", "label", "prediction").show()
""", language="python")
st.write("Making predictions on the test data using the trained model.")

# 5.7 Evaluation
st.subheader("5.7 Evaluation")
st.code("""
# Evaluate model performance
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

evaluator = MulticlassClassificationEvaluator(labelCol="label", predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})
f1_score = evaluator.evaluate(predictions, {evaluator.metricName: "f1"})
mse = BinaryClassificationEvaluator(rawPredictionCol="rawPrediction", labelCol="label", metricName="areaUnderROC").evaluate(predictions)
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})

total_instances = predictions.count()
incorrect_instances = predictions.filter(predictions.label != predictions.prediction).count()
error_rate = incorrect_instances / total_instances

print(f"Accuracy: {accuracy}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1-Score: {f1_score}")
print(f"Root Mean Squared Error: {mse ** 0.5}")
print(f"Error Rate: {error_rate}")
""", language="python")

# Display Evaluation Results
st.write("### Model Evaluation Output")
st.write("""
- **Accuracy:** 0.9968021927820923  
- **Precision:** 0.9968117629085269  
- **Recall:** 0.9968021927820923  
- **F1-Score:** 0.9968017574621731  
- **Root Mean Squared Error:** 0.9999857623288803  
- **Error Rate:** 0.0031978072179077205
""")

# Explanation of Evaluation Metrics
st.write("### Explanation of Evaluation Metrics")
st.write("""
1. **Accuracy**: Measures the percentage of correct predictions out of all predictions. A high accuracy (0.9968) indicates the model performs exceptionally well.
2. **Precision**: Reflects how many of the predicted positive results were truly positive. A precision of 0.9968 indicates very few false positives.
3. **Recall**: Measures the ability of the model to capture all positive cases (true positives). Here, recall is also 0.9968, which indicates most positive cases were detected.
4. **F1-Score**: The harmonic mean of precision and recall. At 0.9968, it confirms the model balances precision and recall effectively.
5. **Root Mean Squared Error (RMSE)**: A measure of the model's prediction error magnitude. A low RMSE (0.9999) shows minimal prediction errors.
6. **Error Rate**: Measures the proportion of incorrect predictions. The small error rate (0.0032) reinforces the model's high performance.
""")

# Final Summary
st.write("""
**Summary**:  
The logistic regression model demonstrates excellent predictive capabilities, with near-perfect accuracy, precision, and recall. This indicates the model generalizes well to unseen data and is suitable for production use.
""")

# VI. UI
st.markdown("<a name='section5'></a>", unsafe_allow_html=True)
styled_header("VI. Application")

# 6.1 Predict function
st.subheader("**`predict_churn(customer_data)` Function**")
st.code( """ def predict_churn(customer_data):
# Preprocess customer data (e.g., feature scaling, encoding)
processed_data = preprocess_data(customer_data)
# Make prediction using the trained model
prediction = trained_model.predict(processed_data)
# Determine churn probability
churn_probability = prediction[0]
return churn_probability
""", language="python" )
st.write( """ * **Data Preprocessing:** Transforms the raw customer data into a format suitablefor
the machine learning model. This may involve scaling numerical features, encoding
categorical variables, and handling missing values. * **Model Prediction:** Utilizes a pre-trained machine learning model (e.g., a logisticregression model, a decision tree, or a neural network) to predict the probability of the
customer churning. * **Probability Calculation:** Extracts the predicted churn probability fromthe model'soutput. """ )
# 6.2 main function
st.subheader("**`main()` Function**")
st.code(""" def main():
st.title("Customer Churn Prediction")
# User input fields for customer data
customer_age = st.number_input("Customer Age", min_value=18)
monthly_bill = st.number_input("Monthly Bill")
customer_tenure = st.number_input("Customer Tenure (months)")
# ... other relevant input fields ... if st.button("Predict Churn"):
customer_data = {
"age": customer_age, "monthly_bill": monthly_bill, "tenure": customer_tenure, # ... other data fields ... }
churn_probability = predict_churn(customer_data)
st.write(f"Predicted Churn Probability: {churn_probability:.2%}")
if churn_probability > 0.5:
st.warning("High risk of customer churn.")
else:
st.success("Low risk of customer churn.") """, language="python" )

st.image(r"pic\GUI1.png")
st.image(r"pic\GUI2.png")
st.write(""" * **User Interface:** Creates an interactive interface using Streamlit components like`st.number_input` to allow users to input customer data. * **Prediction Trigger:** Initiates the churn prediction process when the "Predict
Churn" button is clicked. * **Result Display:** Displays the predicted churn probability in a clear and conciseformat. * **Risk Assessment:** Provides a visual cue (warning/success) to quickly conveythelevel of churn risk. """)

# 6.3 functionality
st.subheader("Functionality")
st.write(""" * **Data Input:** Users provide customer data through interactive input fields. * **Churn Prediction:** The app predicts the likelihood of a customer churning basedon the provided data and the underlying machine learning model.
* **Risk Assessment:** The app provides a visual cue (warning/success) to indicatethelevel of churn risk, enabling users to quickly identify customers requiring immediate attention. """)
# 6.4 conclusion
st.subheader("Remark")
st.write( """This Streamlit app demonstrates a practical application of machine learningforcustomer churn prediction. By providing valuable insights and actionable recommendations, the app can assist businesses in improving customer retention and overall profitability. """)

# 7 Conclusion:
st.markdown("<a name='section6'></a>", unsafe_allow_html=True)
styled_header("VII. Conclusion")
st.write("This project demonstrated the effective use of Spark's MLlib for scalable machine learning on big data. By implementing logistic regression, we highlighted Spark's ability to handle large datasets efficiently. These techniques provided valuable insights and predictions, showcasing the potential for real-world applications.")
st.write("Through this project, we learned how to process and analyze big data using distributed computing, build and evaluate machine learning models at scale, and interpret results to derive actionable insights. Future work could explore streaming data integration and advanced model optimization.")

# 8 References:
reference = {
    1 : "1. [Moro et al., 2014] S. Moro, P. Cortez and P. Rita. A Data-Driven Approach to Predict the Success of Bank Telemarketing. Decision Support Systems, Elsevier, 62:22-31, June 2014.",
    2 : "2. https://www.kaggle.com/datasets/janiobachmann/bank-marketing-dataset/data",
    3 : "3. https://timo.vn/tai-khoan-tiet-kiem/khi-nao-nen-su-dung-goal-save-va-term-deposit/",
    4 : "4. https://spark.apache.org",
    5 : "5. https://www.geeksforgeeks.org/understanding-logistic-regression/"
}
st.markdown("<a name='references'></a>", unsafe_allow_html=True)
styled_header("VIII. References")
st.write(reference[1])
st.write(reference[2])
st.write(reference[3])
st.write(reference[4])
st.write(reference[5])