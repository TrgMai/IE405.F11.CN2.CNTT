import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from utils import load_available_models, get_marketing_strategy
from pyspark.ml import PipelineModel

# --- CẤU HÌNH DỮ LIỆU ---
DEFAULT_VALS = {
    "Age": 39,
    "AnnualIncome": 60,
    "SpendingScore": 50
}

TRAIN_LIMITS = {
    "Age": (18, 70),
    "AnnualIncome": (15, 137),
    "SpendingScore": (1, 99)
}

def show_page(spark):
    st.subheader("🔮 Dự đoán Phân khúc Khách hàng")

    # 1. Load Model
    models = load_available_models()
    if not models:
        st.warning("⚠️ Chưa có model nào. Vui lòng sang tab 'Huấn luyện' để tạo model.")
        return

    col_sel, col_info = st.columns([3, 1])
    with col_sel:
        model_options = {f"{m['name']} (K={m['k']})": m for m in models}
        selected_option = st.selectbox("📂 Chọn Model:", list(model_options.keys()))
        selected_meta = model_options[selected_option]
    with col_info:
        st.write("")
        st.caption(f"Nguồn: {selected_meta['source']}")

    st.divider()

    # 2. Form nhập liệu
    st.markdown("#### 📝 Nhập liệu & Tùy chọn")
    st.caption("ℹ️ *Tích vào ô kiểm để nhập giá trị. Bỏ tích để dùng giá trị mặc định.*")

    with st.form("predict_form"):
        c_basic1, c_basic2 = st.columns(2)
        with c_basic1:
            cid = st.number_input("Mã Khách hàng (ID)", value=1, min_value=1)
        with c_basic2:
            gender = st.selectbox("Giới tính", ["Male", "Female"])
        
        st.write("---")
        
        c1, c2, c3 = st.columns(3)
        
        # Age
        with c1:
            use_age = st.checkbox("Dùng Tuổi (Age)?", value=True)
            if use_age:
                age = st.number_input("Tuổi:", 0, 120, 30)
            else:
                age = DEFAULT_VALS["Age"]
                st.info(f"Giả lập: {age} tuổi")

        # Income
        with c2:
            use_income = st.checkbox("Dùng Thu nhập (Income)?", value=True)
            if use_income:
                income = st.number_input("Income (k$):", 0, 500, 60)
            else:
                income = DEFAULT_VALS["AnnualIncome"]
                st.info(f"Giả lập: {income}k$")

        # Score
        with c3:
            use_score = st.checkbox("Dùng Điểm (Score)?", value=True)
            if use_score:
                score = st.number_input("Score (1-100):", 0, 200, 50)
            else:
                score = DEFAULT_VALS["SpendingScore"]
                st.info(f"Giả lập: {score} điểm")

        st.write("")
        submit = st.form_submit_button("🚀 Phân tích ngay", use_container_width=True)

    # 3. Xử lý Logic
    if submit:
        # A. Validation
        warnings = []
        if use_age and (age < TRAIN_LIMITS["Age"][0] or age > TRAIN_LIMITS["Age"][1]):
            warnings.append(f"⚠️ **Tuổi {age}** nằm ngoài phạm vi ({TRAIN_LIMITS['Age'][0]}-{TRAIN_LIMITS['Age'][1]}).")
        if use_income and (income < TRAIN_LIMITS["AnnualIncome"][0] or income > TRAIN_LIMITS["AnnualIncome"][1]):
            warnings.append(f"⚠️ **Thu nhập {income}k$** nằm ngoài phạm vi ({TRAIN_LIMITS['AnnualIncome'][0]}-{TRAIN_LIMITS['AnnualIncome'][1]}).")
        if use_score and (score < TRAIN_LIMITS["SpendingScore"][0] or score > TRAIN_LIMITS["SpendingScore"][1]):
            warnings.append(f"⚠️ **Điểm {score}** nằm ngoài phạm vi ({TRAIN_LIMITS['SpendingScore'][0]}-{TRAIN_LIMITS['SpendingScore'][1]}).")
        
        if warnings:
            for w in warnings:
                st.warning(w)

        # --- B. XỬ LÝ VỚI SPINNER---
        with st.spinner("⏳ Đang tải model và phân tích dữ liệu... Vui lòng chờ..."):
            
            # 1. Xác định đường dẫn
            if "paths" in selected_meta and "hdfs" in selected_meta["paths"]:
                model_path = selected_meta["paths"]["hdfs"]
            else:
                model_path = f"models/{selected_meta['name']}"

            try:
                # 2. Load Model & Predict
                loaded_model = PipelineModel.load(model_path)
                
                data = [(cid, gender, age, income, score)]
                cols = ["CustomerID", "Gender", "Age", "AnnualIncome", "SpendingScore"]
                df_input = spark.createDataFrame(data, cols)
                
                pred = loaded_model.transform(df_input)
                cluster = pred.select("prediction").collect()[0][0]
                
                # 3. Lấy chiến lược
                name, desc, action = get_marketing_strategy(income, score)
                
                # --- HIỂN THỊ KẾT QUẢ ---
                st.success("✅ Phân tích hoàn tất!")
                
                res_c1, res_c2 = st.columns([1, 1])
                
                with res_c1:
                    st.markdown(f"### Kết quả: Nhóm {cluster}")
                    st.metric(label="Chiến lược đề xuất", value=name)
                    with st.expander("📄 Chi tiết chiến lược", expanded=True):
                        st.info(f"**Đặc điểm:** {desc}")
                        st.write(f"**Hành động:** {action}")

                with res_c2:
                    st.markdown("##### 📍 Vị trí trên bản đồ")
                    fig, ax = plt.subplots(figsize=(6, 5))
                    
                    # Vùng training
                    rect = plt.Rectangle(
                        (TRAIN_LIMITS["AnnualIncome"][0], TRAIN_LIMITS["SpendingScore"][0]), 
                        TRAIN_LIMITS["AnnualIncome"][1] - TRAIN_LIMITS["AnnualIncome"][0],
                        TRAIN_LIMITS["SpendingScore"][1] - TRAIN_LIMITS["SpendingScore"][0],
                        linewidth=1, edgecolor='gray', facecolor='lightgray', alpha=0.3, label='Vùng đã học'
                    )
                    ax.add_patch(rect)
                    
                    # Điểm khách hàng
                    ax.scatter([income], [score], color='#FF4B4B', s=200, marker='*', zorder=5, label='Khách hiện tại')
                    ax.text(income+2, score+2, f"ID: {cid}", fontsize=9, color='darkred', fontweight='bold')

                    ax.set_xlabel("Thu nhập (k$)")
                    ax.set_ylabel("Điểm chi tiêu (1-100)")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.legend(loc='upper right')
                    ax.set_xlim(0, max(160, income + 20))
                    ax.set_ylim(0, max(120, score + 20))
                    
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"Lỗi hệ thống: {e}")