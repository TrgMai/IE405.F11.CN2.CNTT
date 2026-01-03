import streamlit as st

def show_page():
    # --- HEADER ---
    st.title("📑 Đề tài 14: Phân cụm khách hàng")
    st.caption("Môn học: **IE405.F11 - Big Data Applications**")
    st.divider()
    
    # Chia layout: Cột nội dung và Cột thành viên
    col1, col2 = st.columns([2.2, 1], gap="medium")
    
    with col1:
        # 1. Mục tiêu
        st.subheader("🎯 Mục tiêu Dự án")
        st.markdown("""
        Xây dựng hệ thống **Customer Segmentation (Phân khúc khách hàng)** tự động dựa trên hành vi mua sắm, 
        sử dụng kiến trúc Big Data phân tán để hỗ trợ doanh nghiệp đưa ra các chiến lược Marketing cá nhân hóa hiệu quả.
        """)
        
        # 2. Tech Stack (Trình bày dạng bảng cho đẹp)
        st.subheader("🛠 Công nghệ & Kiến trúc")
        with st.container(border=True):
            st.markdown("""
            | Thành phần | Công nghệ sử dụng |
            |---|---|
            | **Processing** | 🔥 Apache Spark 3.4.1 (PySpark) |
            | **Storage** | 🐘 HDFS (Hadoop Distributed File System) |
            | **Model** | 🤖 K-Means Clustering (Spark MLlib) |
            | **Interface** | 🎨 Streamlit Framework |
            | **Architecture** | 🪟 Hybrid (Windows Driver ↔ WSL Workers) |
            """)

        # 3. Quy trình (Pipeline)
        st.subheader("🚀 Luồng xử lý (Pipeline)")
        st.success("""
        1. **Data Ingestion:** Nạp dữ liệu giao dịch khách hàng (CSV/Database).
        2. **Preprocessing:** Làm sạch, xử lý ngoại lai & chuẩn hóa vector (Spark ML).
        3. **Training:** Huấn luyện mô hình K-Means & đánh giá (Tab Training).
        4. **Inference:** Dự đoán phân khúc cho khách hàng mới (Tab Inference).
        """)
    
    with col2:
        st.subheader("👥 Thành viên Nhóm")
        
        members = [
            {
                "name": "Nguyễn Hà My", 
                "id": "24210050", 
                "avatar": "https://img.icons8.com/?size=100&id=7ZVNfAUejd1o&format=png&color=000000" 
            },
            {
                "name": "Trương Mai", 
                "id": "24210046", 
                "avatar": "https://img.icons8.com/?size=100&id=mPPaOMmbhDu6&format=png&color=000000"
            },
            {
                "name": "Lê Ngọc Thuỷ Tiên", 
                "id": "24210087", 
                "avatar": "https://img.icons8.com/?size=100&id=sSe3Hd3iJIK5&format=png&color=000000"
            },
            {
                "name": "Trần Thị Thuỳ Duyên", 
                "id": "24210019", 
                "avatar": "https://img.icons8.com/?size=100&id=1AbMyprPgyuV&format=png&color=000000"
            },
            {
                "name": "Trần Thị Hương Giang", 
                "id": "24210020", 
                "avatar": "https://img.icons8.com/?size=100&id=jgOlIs3QW9z2&format=png&color=000000"
            },
        ]
        
        with st.container(border=True):
            for m in members:
                c_avt, c_info = st.columns([1, 2.5], vertical_alignment="center")
                
                with c_avt:
                    st.image(m["avatar"], width=77) 
                
                with c_info:
                    st.markdown(f"#### {m['name']}")
                    st.caption(f"MSSV: `{m['id']}`")
                
                if m != members[-1]:
                    st.divider()

    st.write("")
    st.info("💡 **Hướng dẫn:** Chuyển sang tab **'🔮 Dự đoán'** để sử dụng mô hình có sẵn hoặc **'🏋️ Huấn luyện'** để train mô hình mới.")