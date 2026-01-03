import sys
import os

# Kiểm tra hệ điều hành
IS_WINDOWS = sys.platform.startswith('win')

# Cấu hình đường dẫn
if IS_WINDOWS:
    # --- CẤU HÌNH CHO WINDOWS (LOCAL) ---
    SPARK_PATH = "C:/spark/spark-3.4.1-bin-hadoop3"
    JAVA_PATH = "C:/Progra~1/Java/jdk-11"
    HADOOP_PATH = "C:/spark/spark-3.4.1-bin-hadoop3/hadoop"
    
    from config_ip import get_windows_ip, get_wsl_ip
    WINDOWS_IP = get_windows_ip()
    WSL_IP = get_wsl_ip()
    
    SPARK_MASTER = f"spark://{WINDOWS_IP}:7077"
    HDFS_URL = f"hdfs://{WSL_IP}:9000"
    
else:
    # --- CẤU HÌNH CHO LINUX (STREAMLIT CLOUD) ---
    SPARK_PATH = None 
    JAVA_PATH = None
    HADOOP_PATH = None
    
    WINDOWS_IP = "127.0.0.1"
    WSL_IP = "127.0.0.1"
    
    SPARK_MASTER = "local[*]"
    HDFS_URL = None