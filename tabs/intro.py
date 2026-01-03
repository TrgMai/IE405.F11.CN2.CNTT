import streamlit as st

def show_page():
    st.header("Thông tin Dự án")
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown("""
        ### 🎯 Mục tiêu
        Xây dựng hệ thống **Customer Segmentation** tự động dựa trên hành vi mua sắm, sử dụng kiến trúc Big Data phân tán.
        
        ### 🛠 Công nghệ
        * **Processing:** Apache Spark 3.4.1 (PySpark)
        * **Storage:** HDFS (Hadoop Distributed File System)
        * **Model:** K-Means Clustering (Spark MLlib)
        * **Interface:** Streamlit
        """)
    
    with col2:
        st.info("### 👥 Thành viên Nhóm")
        st.write("""
        **1. Trương Mai** (MSSV: ...)
        *Vai trò: Data Engineer, Spark Setup*
        
        **2. Thành viên B**
        *Vai trò: Data Analyst*
        
        **3. Thành viên C**
        *Vai trò: Frontend Dev*
        """)
    
    st.divider()
    st.caption("Đồ án môn học IE405.F11 - Big Data Applications")