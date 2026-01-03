import streamlit as st
import os
import sys
import socketserver

if not hasattr(socketserver, "UnixStreamServer"):
    socketserver.UnixStreamServer = socketserver.TCPServer

from config_env import IS_WINDOWS, SPARK_MASTER, WINDOWS_IP, SPARK_PATH, JAVA_PATH, HADOOP_PATH

if IS_WINDOWS:
    os.environ['SPARK_HOME'] = SPARK_PATH
    os.environ['JAVA_HOME'] = JAVA_PATH
    os.environ['HADOOP_HOME'] = HADOOP_PATH
    os.environ['PYSPARK_DRIVER_PYTHON'] = sys.executable
    os.environ['PYSPARK_PYTHON'] = '/usr/bin/python3.10'
    os.environ["HADOOP_USER_NAME"] = "ubt_trgmai"
    
    sys.path.insert(0, os.path.join(SPARK_PATH, "python"))
    sys.path.insert(0, os.path.join(SPARK_PATH, "python", "lib", "py4j-0.10.9.7-src.zip"))

from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from tabs import intro, inference, training

@st.cache_resource
def init_spark():
    conf = SparkConf().setAppName("Marketing_App_UI").setMaster(SPARK_MASTER)
    
    if IS_WINDOWS:
        conf.set("spark.driver.host", WINDOWS_IP) \
            .set("spark.driver.bindAddress", WINDOWS_IP) \
            .set("spark.executor.memory", "1g") \
            .set("spark.cores.max", "2") \
            .set("spark.rpc.message.maxSize", "1024") \
            .set("spark.kryoserializer.buffer.max", "1024m")
    else:
        conf.set("spark.driver.memory", "2g") \
            .set("spark.executor.memory", "1g")
            
    return SparkSession.builder.config(conf=conf).getOrCreate()

st.set_page_config(page_title="Big Data Marketing", layout="wide", page_icon="🛍️")

env_mode = f"Driver (Windows: {WINDOWS_IP}) <--> Worker (WSL Cluster)" if IS_WINDOWS else "Standalone (Streamlit Cloud Linux)"
st.title("🛍️ Hệ thống Phân nhóm Khách hàng (Big Data)")
st.caption(f"🚀 **Environment:** {env_mode}")

try:
    spark = init_spark()
    st.toast(f"✅ Spark Connected! Ver: {spark.version}", icon="🔥")
except Exception as e:
    st.error(f"❌ Lỗi kết nối Spark: {e}")
    st.stop()

tab1, tab2, tab3 = st.tabs(["ℹ️ Giới thiệu & Nhóm", "🔮 Dự đoán (Inference)", "🏋️ Huấn luyện (Training)"])

with tab1: intro.show_page()
with tab2: inference.show_page(spark)
with tab3: training.show_page(spark)