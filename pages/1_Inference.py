import streamlit as st
from tools.utils import load_available_models, get_marketing_strategy
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession

st.header("🔮 Dự đoán Phân khúc Khách hàng")

# 1. Load danh sách Model
models = load_available_models()
if not models:
    st.warning("Chưa có model nào. Vui lòng Train ở Notebook hoặc trang Training.")
    st.stop()

# Dropdown chọn model
model_options = {f"{m['name']} (K={m['k']}, Src={m['source']})": m for m in models}
selected_option = st.selectbox("Chọn Model:", list(model_options.keys()))
selected_meta = model_options[selected_option]

# 2. Form nhập liệu
col1, col2 = st.columns(2)
with col1:
    cid = st.number_input("Customer ID", value=1)
    gender = st.selectbox("Gender", ["Male", "Female"])
    age = st.number_input("Age", 18, 100, 30)
with col2:
    income = st.number_input("Annual Income (k$)", 10, 200, 50)
    score = st.number_input("Spending Score (1-100)", 1, 100, 50)

if st.button("Phân tích"):
    # Load Model Spark
    spark = SparkSession.builder.getOrCreate()

    if "paths" in selected_meta and "hdfs" in selected_meta["paths"]:
        model_path = selected_meta["paths"]["hdfs"]
        print(f"Debug: Loading HDFS path: {model_path}")
    else:
        model_path = f"models/{selected_meta['name']}"

    loaded_model = PipelineModel.load(model_path)
    
    # Tạo DataFrame
    data = [(cid, gender, age, income, score)]
    cols = ["CustomerID", "Gender", "Age", "AnnualIncome", "SpendingScore"]
    df_input = spark.createDataFrame(data, cols)
    
    # Predict
    pred = loaded_model.transform(df_input)
    cluster = pred.select("prediction").collect()[0][0]
    
    # Marketing Rule
    name, desc, action = get_marketing_strategy(income, score)
    
    # Display
    st.success(f"Khách hàng thuộc Cụm: {cluster}")
    st.markdown(f"### 🎯 Chiến lược: {name}")
    st.write(f"**Đặc điểm:** {desc}")
    st.info(f"**Hành động:** {action}")
    st.caption(f"Dự đoán bởi model: {selected_meta['name']} (Train date: {selected_meta['date']})")