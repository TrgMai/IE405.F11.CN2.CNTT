import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tools.utils import load_available_models, get_marketing_strategy, load_model_smart
from pyspark.ml import PipelineModel
import traceback
import numpy as np
import os

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
        if st.button("🔄 Kiểm tra lại"):
            st.rerun()
        return

    c_sel, c_info, c_btn = st.columns([3, 1.2, 1], vertical_alignment="bottom", gap="small")
    
    with c_sel:
        model_options = {f"{m['name']} (K={m['k']})": m for m in models}
        selected_option = st.selectbox("📂 Chọn Model:", list(model_options.keys()))
        selected_meta = model_options[selected_option]
        
    with c_info:
        st.caption("Nguồn dữ liệu:")
        source_label = "☁️ HDFS" if "HDFS" in selected_meta['source'] else "💻 Local"
        st.markdown(f"**{source_label}**")

    with c_btn:
        if st.button("🔄 Cập nhật Model", help="Cập nhật danh sách model mới nhất"):
            st.rerun()

    st.divider()

    # --- CẤU HÌNH LƯU TRỮ FILE CSV ---
    DATA_DIR = "data"
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    # Đường dẫn file cố định để lưu đè mỗi lần upload
    CSV_STORAGE_PATH = os.path.join(DATA_DIR, "inference_customers.csv")

    # --- 2. GIAO DIỆN NHẬP LIỆU (CHIA 2 PHẦN) ---
    tab_list, tab_manual = st.tabs(["📂 Chọn khách hàng từ danh sách", "📝 Nhập liệu thủ công"])
    
    # Biến để hứng dữ liệu
    input_data = None
    submit = False

    # === PHẦN 1: CHỌN TỪ DANH SÁCH (CÓ LƯU FILE) ===
    with tab_list:
        st.markdown("#### 📋 Chọn khách hàng có sẵn")
        
        # Logic Upload & Lưu File
        uploaded_file = st.file_uploader("Upload danh sách (CSV) - *Sẽ tự động lưu cho lần sau*", type=["csv"], key="infer_upload")
        
        df_cust = None
        
        # Nếu có upload mới -> Lưu đè file cũ
        if uploaded_file is not None:
            try:
                # Lưu file
                with open(CSV_STORAGE_PATH, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                st.toast("✅ Đã lưu danh sách mới!", icon="💾")
            except Exception as e:
                st.error(f"❌ Lỗi khi lưu file: {e}")

        # Kiểm tra và Load file từ bộ nhớ (Disk)
        if os.path.exists(CSV_STORAGE_PATH):
            try:
                df_cust = pd.read_csv(CSV_STORAGE_PATH)
                if uploaded_file is None:
                    st.info(f"📂 Đang sử dụng danh sách đã lưu từ trước ({len(df_cust)} khách hàng).")
            except Exception as e:
                st.error(f"❌ File đã lưu bị lỗi, vui lòng upload lại. Lỗi: {e}")
                # Nếu lỗi thì xóa file hỏng đi
                os.remove(CSV_STORAGE_PATH)
        else:
            if uploaded_file is None:
                st.warning("⚠️ Chưa có dữ liệu. Vui lòng Upload file CSV lần đầu tiên.")

        # Xử lý hiển thị nếu đã có dữ liệu (từ file vừa upload hoặc file cũ)
        if df_cust is not None:
            try:
                # Chuẩn hóa tên cột
                df_cust = df_cust.rename(columns={
                    "Annual Income (k$)": "AnnualIncome",
                    "Spending Score (1-100)": "SpendingScore"
                })

                # Kiểm tra cột ID
                if "CustomerID" in df_cust.columns:
                    cust_ids = df_cust["CustomerID"].tolist()
                    selected_id = st.selectbox("Chọn Mã Khách hàng (ID):", cust_ids)
                    row = df_cust[df_cust["CustomerID"] == selected_id].iloc[0]
                else:
                    st.warning("⚠️ File thiếu cột 'CustomerID'. Dùng số thứ tự dòng.")
                    selected_index = st.selectbox("Chọn STT Khách hàng:", df_cust.index)
                    row = df_cust.iloc[selected_index]
                    selected_id = selected_index

                # Hiển thị thông tin
                st.write("**Thông tin khách hàng:**")
                d_c1, d_c2, d_c3, d_c4 = st.columns(4)
                d_c1.metric("Giới tính", row.get("Gender", "Unknown"))
                d_c2.metric("Tuổi", row.get("Age", 0))
                d_c3.metric("Thu nhập", f"{row.get('AnnualIncome', 0)} k$")
                d_c4.metric("Điểm số", row.get("SpendingScore", 0))
                
                # Nút phân tích
                if st.button("🚀 Phân tích khách hàng này", use_container_width=True, key="btn_list"):
                    # Ép kiểu dữ liệu Python chuẩn (int/float)
                    input_data = {
                        "cid": int(selected_id) if isinstance(selected_id, (int, np.integer)) else str(selected_id),
                        "gender": str(row.get("Gender", "Male")),
                        "age": int(row.get("Age", DEFAULT_VALS["Age"])),
                        "income": float(row.get("AnnualIncome", DEFAULT_VALS["AnnualIncome"])),
                        "score": float(row.get("SpendingScore", DEFAULT_VALS["SpendingScore"])),
                        "use_age": True, "use_income": True, "use_score": True
                    }
                    submit = True
            except Exception as e:
                st.error(f"❌ Lỗi xử lý dữ liệu: {e}")

    # === PHẦN 2: NHẬP THỦ CÔNG (KHÔNG CẦN ID) ===
    with tab_manual:
        st.markdown("#### ✍️ Tự nhập thông số")
        st.caption("ℹ️ *Tích vào ô kiểm để nhập giá trị. Bỏ tích để dùng giá trị mặc định.*")

        with st.form("predict_form_manual"):
            # CHỈ CÒN: Giới tính (Bỏ nhập ID)
            m_gender = st.selectbox("Giới tính", ["Male", "Female"])
            
            st.write("---")
            c1, c2, c3 = st.columns(3)
            
            # Age
            with c1:
                m_use_age = st.checkbox("Dùng Tuổi (Age)?", value=True)
                if m_use_age:
                    m_age = st.number_input("Tuổi:", 0, 120, 30)
                else:
                    m_age = DEFAULT_VALS["Age"]
                    st.info(f"Giả lập: {m_age} tuổi")

            # Income
            with c2:
                m_use_income = st.checkbox("Dùng Thu nhập (Income)?", value=True)
                if m_use_income:
                    m_income = st.number_input("Income (k$):", 0, 500, 60)
                else:
                    m_income = DEFAULT_VALS["AnnualIncome"]
                    st.info(f"Giả lập: {m_income}k$")

            # Score
            with c3:
                m_use_score = st.checkbox("Dùng Điểm (Score)?", value=True)
                if m_use_score:
                    m_score = st.number_input("Score (1-100):", 0, 200, 50)
                else:
                    m_score = DEFAULT_VALS["SpendingScore"]
                    st.info(f"Giả lập: {m_score} điểm")

            st.write("")
            submit_manual = st.form_submit_button("🚀 Phân tích ngay", use_container_width=True)
            
            if submit_manual:
                # Tự động gán ID = 0 cho khách hàng mới nhập tay
                input_data = {
                    "cid": 0,  
                    "gender": m_gender, 
                    "age": int(m_age),
                    "income": float(m_income), 
                    "score": float(m_score),
                    "use_age": m_use_age, "use_income": m_use_income, "use_score": m_use_score
                }
                submit = True

    # --- 3. XỬ LÝ LOGIC CHUNG (SPARK) ---
    if submit and input_data:
        # Unpack dữ liệu
        cid = input_data["cid"]
        gender = input_data["gender"]
        age = input_data["age"]
        income = input_data["income"]
        score = input_data["score"]
        use_age = input_data["use_age"]
        use_income = input_data["use_income"]
        use_score = input_data["use_score"]

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

        # --- B. XỬ LÝ VỚI SPARK ---
        with st.spinner("⏳ Đang tải model và phân tích dữ liệu... Vui lòng chờ..."):
            
            try:
                # Load model
                loaded_model, source_type = load_model_smart(selected_meta['name'], selected_meta)
                print(f"✅ Loaded model successfully from: {source_type}")

                # Chuẩn bị dữ liệu (đã ép kiểu chuẩn)
                data = [(cid, gender, age, income, score)]
                cols = ["CustomerID", "Gender", "Age", "AnnualIncome", "SpendingScore"]
                
                df_input = spark.createDataFrame(data, cols)
                
                # Dự đoán
                pred = loaded_model.transform(df_input)
                
                # Collect kết quả
                row_result = pred.select("prediction").collect()[0]
                cluster = row_result[0]
                
                # Lấy chiến lược
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
                    
                    # Hiển thị ID (nếu là nhập tay id=0 -> Hiện chữ 'New')
                    label_id = cid if cid != 0 else "New"
                    ax.text(income+2, score+2, f"ID: {label_id}", fontsize=9, color='darkred', fontweight='bold')

                    ax.set_xlabel("Thu nhập (k$)")
                    ax.set_ylabel("Điểm chi tiêu (1-100)")
                    ax.grid(True, linestyle='--', alpha=0.5)
                    ax.legend(loc='upper right')
                    ax.set_xlim(0, max(160, income + 20))
                    ax.set_ylim(0, max(120, score + 20))
                    
                    st.pyplot(fig)

            except Exception as e:
                st.error(f"❌ Lỗi hệ thống: {e}")
                traceback.print_exc()