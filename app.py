import streamlit as st
import numpy as np
import tensorflow as tf
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler
from langchain_huggingface import HuggingFaceEndpoint
from langchain import PromptTemplate, LLMChain

# 🎯 Load trained model (TensorFlow or Scikit-Learn)
try:
    model = tf.keras.models.load_model("breast_cancer_model.h5")
    model_type = "tensorflow"
except:
    model = joblib.load("model.pkl")
    model_type = "sklearn"

# 📊 Load dataset for feature scaling
df = pd.read_csv("breast_cancer.csv")
X_train = df.drop(columns=["diagnosis", "id", "Unnamed: 32"], errors="ignore")

# 📏 Load or fit StandardScaler
try:
    scaler = joblib.load("scaler.pkl")
except:
    scaler = StandardScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, "scaler.pkl")

# 🔐 Securely Load Hugging Face API Key
sec_key = st.secrets["huggingface"]["api_key"]
repo_id = "mistralai/Mistral-7B-Instruct-v0.3"

# 🔗 Initialize AI Model
llm = HuggingFaceEndpoint(repo_id=repo_id, max_length=128, temperature=0.7, token=sec_key)

# 🏥 **App Title**
st.title("🔬 Breast Cancer Detection Assistant")

# 📝 **Instructions**
st.markdown("Enter the tumor measurements below to check if it’s **Benign** or **Malignant**.")

# 📌 **Sidebar - Feature Input**
st.sidebar.header("🔢 Enter Tumor Features")
feature_names = X_train.columns.tolist()

# 📌 Default values (mean of dataset)
default_values = [
    11.76, 21.6, 74.72, 427.9, 0.08637, 0.04966, 0.01657, 0.01115, 0.1495, 0.05888,
    0.4062, 1.21, 2.635, 28.47, 0.005857, 0.009758, 0.01168, 0.007445, 0.02406, 0.001769,
    12.98, 25.72, 82.98, 516.5, 0.1085, 0.08615, 0.05523, 0.03715, 0.2433, 0.06563
]

# 🎛 Take user inputs dynamically
user_inputs = []
for feature, default in zip(feature_names, default_values):
    value = st.sidebar.number_input(f"{feature.replace('_', ' ').title()}", value=float(default))
    user_inputs.append(value)

# 🩺 **Predict Button**
if st.button("🩺 Get Prediction", type="primary"):
    try:
        input_values = np.array(user_inputs).reshape(1, -1)
        input_values = scaler.transform(input_values)

        if model_type == "tensorflow":
            prediction = model.predict(input_values)
            prediction_label = np.argmax(prediction, axis=1)[0]
        else:
            prediction_label = model.predict(input_values)[0]

        st.markdown("---")
        if prediction_label == 0:
            st.success("✅ **Result: Likely Benign**\nThis tumor is likely **non-cancerous**.")
        else:
            st.error("⚠ **Result: Likely Malignant**\nThis tumor is likely **cancerous**. Please consult a doctor.")
    except Exception as e:
        st.error(f"Error: {str(e)}")

# 🤖 **Chatbot Section**
st.markdown("---")
st.header("🗨 Breast Cancer Chatbot")
st.write("Ask any question about breast cancer and get AI-powered answers.")

user_question = st.text_input("Type your question here:")

if st.button("Ask AI", type="primary") and user_question:
    try:
        template = """Question: {question}\n\nAnswer: Let's think step by step."""
        prompt = PromptTemplate(template=template, input_variables=["question"])
        llm_chain = LLMChain(llm=llm, prompt=prompt)
        response = llm_chain.invoke(user_question)

        st.write("### 🤖 AI Response:")
        st.write(response)
    except Exception as e:
        st.error(f"Error: {str(e)}")
