import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import build_pipeline, save_model_hybrid, get_marketing_strategy

def show_page(spark):
    st.subheader("🏋️ Huấn luyện Mô hình Mới (Analysis & Training)")

    # 1. Upload Data
    uploaded_file = st.file_uploader("📂 Upload CSV Data (để train lại)", type=["csv"])

    if uploaded_file:
        pdf = pd.read_csv(uploaded_file)
        
        # --- A. PHÂN TÍCH DỮ LIỆU ĐẦU VÀO ---
        with st.expander("📊 Phân tích Dữ liệu Chuyên sâu (Exploratory Data Analysis)", expanded=True):
            st.write("#### 1. Phân bố Dữ liệu (Distribution)")
            
            # Vẽ 3 biểu đồ Histogram cho Age, Income, Score
            c1, c2, c3 = st.columns(3)
            with c1:
                st.caption("Phân bố Tuổi (Age)")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(pdf['Age'], kde=True, color='teal', bins=20, ax=ax)
                st.pyplot(fig)
            with c2:
                st.caption("Phân bố Thu nhập (Income)")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(pdf['Annual Income (k$)'], kde=True, color='skyblue', bins=20, ax=ax)
                st.pyplot(fig)
            with c3:
                st.caption("Phân bố Điểm số (Score)")
                fig, ax = plt.subplots(figsize=(4, 3))
                sns.histplot(pdf['Spending Score (1-100)'], kde=True, color='salmon', bins=20, ax=ax)
                st.pyplot(fig)

            st.write("#### 2. Tương quan & Ngoại lai")
            c_corr, c_box = st.columns(2)
            
            # Ma trận tương quan (Heatmap)
            with c_corr:
                st.caption("Ma trận Tương quan (Correlation Heatmap)")
                fig, ax = plt.subplots(figsize=(5, 4))
                # Lọc chỉ lấy cột số
                numeric_df = pdf[['Age', 'Annual Income (k$)', 'Spending Score (1-100)']]
                sns.heatmap(numeric_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
                st.pyplot(fig)
            
            # Biểu đồ hộp (Boxplot) tìm Outlier
            with c_box:
                st.caption("Kiểm tra Ngoại lai (Boxplot)")
                fig, ax = plt.subplots(figsize=(5, 4))
                sns.boxplot(data=numeric_df, orient="h", palette="Set2", ax=ax)
                st.pyplot(fig)

        # Convert to Spark DataFrame
        sdf = spark.createDataFrame(pdf)
        sdf = sdf.withColumnRenamed("Annual Income (k$)", "AnnualIncome") \
                 .withColumnRenamed("Spending Score (1-100)", "SpendingScore")

        st.divider()

        # --- B. CẤU HÌNH & TRAIN ---
        st.markdown("#### ⚙️ Cấu hình Huấn luyện")
        c_conf1, c_conf2, c_btn = st.columns([2, 2, 1])
        with c_conf1:
            k = st.slider("Số cụm (K - Clusters)", 2, 10, 5)
        with c_conf2:
            model_name_input = st.text_input("Đặt tên Model", f"ui_model_k{k}")
        with c_btn:
            st.write("") 
            st.write("") 
            start_btn = st.button("▶️ Bắt đầu Train", type="primary", use_container_width=True)

        if start_btn:
            with st.spinner(f"⏳ Đang huấn luyện K-Means (K={k}) trên Spark..."):
                try:
                    pipeline = build_pipeline(k)
                    model = pipeline.fit(sdf)
                    predictions = model.transform(sdf)
                    
                    # Convert về Pandas để vẽ biểu đồ
                    result_pdf = predictions.select("AnnualIncome", "SpendingScore", "prediction").toPandas()
                    
                    st.success("🎉 Huấn luyện hoàn tất!")
                    
                    # --- C. PHÂN TÍCH KẾT QUẢ (RESULT ANALYSIS) ---
                    st.markdown("### 🔍 Phân tích Kết quả Phân cụm")
                    
                    # 1. Biểu đồ chính (Scatter Plot & Count Plot)
                    rc1, rc2 = st.columns(2)
                    with rc1:
                        st.markdown("**1. Bản đồ Phân cụm (Income vs Score)**")
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.scatterplot(data=result_pdf, x="AnnualIncome", y="SpendingScore", hue="prediction", palette="viridis", s=100, ax=ax)
                        ax.legend(title="Cluster")
                        st.pyplot(fig)
                    
                    with rc2:
                        st.markdown("**2. Số lượng Khách hàng mỗi Cụm**")
                        fig, ax = plt.subplots(figsize=(6, 5))
                        sns.countplot(x='prediction', data=result_pdf, palette='viridis', ax=ax)
                        ax.set_xlabel("Cluster ID")
                        ax.set_ylabel("Số lượng khách")
                        
                        # Hiển thị số liệu trên đầu cột
                        for p in ax.patches:
                            ax.annotate(f'{int(p.get_height())}', (p.get_x() + p.get_width() / 2., p.get_height()), 
                                        ha='center', va='bottom', fontsize=10, color='black', xytext=(0, 5), textcoords='offset points')
                        st.pyplot(fig)

                    # 2. Phân tích sâu (Boxplots)
                    st.markdown("**3. So sánh Đặc điểm các Cụm (Boxplots)**")
                    bc1, bc2 = st.columns(2)
                    with bc1:
                        st.caption("So sánh Thu nhập giữa các Cụm")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.boxplot(x='prediction', y='AnnualIncome', data=result_pdf, palette='viridis', ax=ax)
                        st.pyplot(fig)
                    with bc2:
                        st.caption("So sánh Điểm chi tiêu giữa các Cụm")
                        fig, ax = plt.subplots(figsize=(6, 4))
                        sns.boxplot(x='prediction', y='SpendingScore', data=result_pdf, palette='viridis', ax=ax)
                        st.pyplot(fig)

                    # 3. Bảng thống kê chi tiết
                    st.markdown("**4. Bảng Tổng hợp Chiến lược**")
                    stats = result_pdf.groupby('prediction').mean(numeric_only=True).reset_index()
                    report = []
                    for _, row in stats.iterrows():
                        inc = row['AnnualIncome']
                        sc = row['SpendingScore']
                        name, _, _ = get_marketing_strategy(inc, sc)
                        report.append({
                            "Cluster": int(row['prediction']), 
                            "Avg Income ($k)": f"{inc:.1f}", 
                            "Avg Score": f"{sc:.1f}", 
                            "Đề xuất Chiến lược": name
                        })
                    
                    st.dataframe(pd.DataFrame(report), use_container_width=True)

                    # Lưu Model
                    save_path = save_model_hybrid(model, model_name_input, k, "ui")
                    st.success(f"💾 Model đã lưu tại: `{save_path}`")
                    st.info("💡 Mẹo: Chuyển sang tab 'Dự đoán' để dùng thử model này.")
                    
                except Exception as e:
                    st.error(f"❌ Lỗi Training: {e}")
                    import traceback
                    traceback.print_exc()