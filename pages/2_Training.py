import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pyspark.sql import SparkSession
from utils import build_pipeline, save_model_hybrid, get_marketing_strategy

st.header("🏋️ Huấn luyện Mô hình Mới")

spark = SparkSession.builder.getOrCreate()

uploaded_file = st.file_uploader("Upload CSV Data", type=["csv"])

if uploaded_file:
    pdf = pd.read_csv(uploaded_file)
    st.write("Preview:", pdf.head())
    
    # EDA
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Phân bố Thu nhập")
        fig1, ax1 = plt.subplots()
        sns.histplot(pdf['Annual Income (k$)'], kde=True, ax=ax1)
        st.pyplot(fig1)
    with col2:
        st.subheader("Income vs Score")
        fig2, ax2 = plt.subplots()
        sns.scatterplot(data=pdf, x='Annual Income (k$)', y='Spending Score (1-100)', ax=ax2)
        st.pyplot(fig2)

    # Convert to Spark
    sdf = spark.createDataFrame(pdf)
    # Rename cols
    sdf = sdf.withColumnRenamed("Annual Income (k$)", "AnnualIncome") \
             .withColumnRenamed("Spending Score (1-100)", "SpendingScore")

    # Training Config
    k = st.slider("Chọn số cụm (K)", 2, 10, 5)
    model_name_input = st.text_input("Tên Model lưu trữ", f"ui_model_k{k}")

    if st.button("Bắt đầu Train (Distributed)"):
        with st.spinner("Worker đang xử lý..."):
            # Dùng Pipeline chung từ Utils
            pipeline = build_pipeline(k)
            model = pipeline.fit(sdf)
            predictions = model.transform(sdf)
            
            # Show Results
            result_pdf = predictions.select("AnnualIncome", "SpendingScore", "prediction").toPandas()
            
            st.subheader("Kết quả Phân cụm")
            fig3, ax3 = plt.subplots()
            sns.scatterplot(data=result_pdf, x="AnnualIncome", y="SpendingScore", hue="prediction", palette="viridis", ax=ax3)
            st.pyplot(fig3)
            
            # Stats & Strategy
            st.subheader("Phân tích & Chiến lược")
            stats = result_pdf.groupby('prediction').mean().reset_index()
            
            report = []
            for _, row in stats.iterrows():
                inc = row['AnnualIncome']
                sc = row['SpendingScore']
                name, _, _ = get_marketing_strategy(inc, sc)
                report.append({"Cluster": row['prediction'], "Avg Income": inc, "Avg Score": sc, "Strategy": name})
            
            st.table(pd.DataFrame(report))

            save_path = save_model_hybrid(model, model_name_input, k, "ui")
            st.success(f"✅ Model đã lưu tại: {save_path}")
            st.info("Bạn có thể sang trang Inference để dùng model này ngay lập tức.")