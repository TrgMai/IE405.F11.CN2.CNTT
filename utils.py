from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import VectorAssembler, StandardScaler
from pyspark.ml.clustering import KMeans
import json
import os
import shutil
import subprocess
from datetime import datetime

# --- IMPORT CẤU HÌNH MÔI TRƯỜNG ---
try:
    from config_env import IS_WINDOWS
except ImportError:
    import sys
    IS_WINDOWS = sys.platform.startswith('win')

if IS_WINDOWS:
    from config_ip import get_wsl_ip

# --- 0. HELPER: TỰ ĐỘNG DÒ TÌM HDFS ---
def detect_hdfs_path():
    if not IS_WINDOWS:
        return None

    # Cách 1: Thử lệnh 'which hdfs'
    try:
        cmd = ["wsl", "bash", "-l", "-c", "which hdfs"]
        path = subprocess.check_output(cmd).decode("utf-8").strip()
        if path and "hdfs" in path:
            return path
    except:
        pass

    # Cách 2: Quét tìm file trong thư mục Home
    try:
        find_cmd = "find ~/ -name hdfs -type f -path '*/bin/hdfs' 2>/dev/null | head -n 1"
        full_cmd = ["wsl", "bash", "-l", "-c", find_cmd]
        path = subprocess.check_output(full_cmd).decode("utf-8").strip()
        if path:
            return path
    except:
        pass

    return "hdfs"

# --- 1. PIPELINE FACTORY ---
def build_pipeline(k):
    assembler = VectorAssembler(inputCols=["AnnualIncome", "SpendingScore"], outputCol="features_raw")
    scaler = StandardScaler(inputCol="features_raw", outputCol="features", withStd=True, withMean=True)
    kmeans = KMeans(k=k, seed=42, featuresCol="features", predictionCol="prediction")
    pipeline = Pipeline(stages=[assembler, scaler, kmeans])
    return pipeline

# --- 2. MARKETING STRATEGY ---
def get_marketing_strategy(income, score):
    if income > 70 and score > 60:
        return "VIP", "Thu nhập CAO - Chi tiêu CAO", "👑 Chăm sóc đặc biệt, upsell hàng hiệu."
    elif income > 70 and score < 40:
        return "Tiềm năng", "Thu nhập CAO - Chi tiêu THẤP", "💼 Gợi ý sản phẩm chất lượng, kích cầu."
    elif income < 40 and score > 60:
        return "Rủi ro", "Thu nhập THẤP - Chi tiêu CAO", "⚠️ Giới thiệu trả góp, khuyến mãi giá rẻ."
    elif income < 40 and score < 40:
        return "Tiết kiệm", "Thu nhập THẤP - Chi tiêu THẤP", "💰 Gửi voucher giảm giá, hàng thiết yếu."
    else:
        return "Tiêu chuẩn", "Khách hàng Trung bình", "📧 Duy trì tương tác định kỳ."

# --- 3. HELPER: CONVERT PATH WINDOWS -> WSL ---
def windows_to_wsl_path(windows_path):
    if not IS_WINDOWS:
        return windows_path # Giữ nguyên nếu là Linux
        
    path = os.path.abspath(windows_path).replace("\\", "/")
    if len(path) > 1 and path[1] == ":":
        drive_letter = path[0].lower() # Lấy chữ cái ổ đĩa (c, d, e...)
        rest_of_path = path[2:]        # Lấy phần còn lại
        return f"/mnt/{drive_letter}{rest_of_path}"
    return path 

# --- 4. HYBRID SAVE (QUAN TRỌNG: TỰ ĐỘNG CHỌN LOGIC LƯU) ---
def save_model_hybrid(model, model_name, k, source, local_root="models"):
    
    # Tạo thư mục local nếu chưa có
    if not os.path.exists(local_root):
        os.makedirs(local_root)

    # --- TRƯỜNG HỢP 1: WINDOWS ---
    if IS_WINDOWS:
        wsl_ip = get_wsl_ip()
        
        # Tự động dò tìm đường dẫn HDFS
        HDFS_BIN = detect_hdfs_path()
        print(f"🔎 Detected HDFS Path: {HDFS_BIN}")
        
        # Định nghĩa đường dẫn
        hdfs_path = f"hdfs://{wsl_ip}:9000/project/models/{model_name}"
        local_path = os.path.abspath(os.path.join(local_root, model_name))
        wsl_local_path = windows_to_wsl_path(local_path) 
        
        # BƯỚC 1: Lưu lên HDFS (Spark Native Write)
        print(f"🔄 [1/3] Saving to HDFS: {hdfs_path}...")
        try:
            model.write().overwrite().save(hdfs_path)
        except Exception as e:
            print(f"❌ Lỗi khi lưu HDFS: {e}")
            raise e
        
        # BƯỚC 2: Copy từ HDFS về Local Windows (Thông qua WSL CLI)
        print(f"🔄 [2/3] Syncing to Local Windows: {local_path}...")
        
        if os.path.exists(local_path):
            try:
                shutil.rmtree(local_path)
            except Exception as e:
                print(f"⚠️ Không thể xóa folder cũ: {e}")
            
        try:
            bash_cmd = f"'{HDFS_BIN}' dfs -get '{hdfs_path}' '{wsl_local_path}'"
            subprocess.check_call(["wsl", "bash", "-l", "-c", bash_cmd])
            print("✅ Sync Local thành công!")
        except Exception as e:
            print(f"⚠️ Lỗi Sync về Windows: {e}")

        # Metadata có đường dẫn HDFS
        meta_paths = { "hdfs": hdfs_path, "local": local_path }

    # --- TRƯỜNG HỢP 2: LINUX/CLOUD ---
    else:
        print(f"☁️ Detect Linux Environment. Saving locally to {local_root}...")
        local_path = os.path.join(local_root, model_name)
        
        # Xóa model cũ nếu có
        if os.path.exists(local_path):
            shutil.rmtree(local_path)
            
        # Lưu trực tiếp bằng Spark (Local FS)
        model.write().overwrite().save(local_path)
        
        # Metadata không có HDFS
        meta_paths = { "hdfs": None, "local": local_path }

    # --- BƯỚC 3: LƯU METADATA JSON ---
    meta = {
        "name": model_name,
        "k": k,
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "source": source,
        "paths": meta_paths
    }
    
    meta_path = os.path.join(local_root, f"{model_name}_meta.json")
    with open(meta_path, "w", encoding='utf-8') as f:
        json.dump(meta, f, ensure_ascii=False, indent=4)
        
    return local_path

# --- 5. SMART LOAD ---
def load_model_smart(model_name, meta=None):
    local_path = os.path.abspath(os.path.join("models", model_name))
    
    # --- Ưu tiên 1: Load từ Local (Windows Folder hoặc Cloud Folder) ---
    if os.path.exists(local_path):
        try:
            # Xử lý đường dẫn cho Spark
            if IS_WINDOWS:
                # Windows cần: file:///C:/path/to/model
                uri_path = "file:///" + local_path.replace("\\", "/").lstrip("/")
            else:
                # Linux cần: file:///path/to/model
                uri_path = "file://" + local_path
                
            print(f"📂 Đang load từ LOCAL: {uri_path}")
            model = PipelineModel.load(uri_path)
            return model, "Local Storage"
        except Exception as e:
            print(f"⚠️ Load Local thất bại ({e}). Đang thử nguồn khác...")
    
    # --- Ưu tiên 2: Load từ HDFS ---
    if IS_WINDOWS:
        if meta and "paths" in meta and meta["paths"]["hdfs"]:
            hdfs_path = meta["paths"]["hdfs"]
        else:
            wsl_ip = get_wsl_ip()
            hdfs_path = f"hdfs://{wsl_ip}:9000/project/models/{model_name}"

        try:
            print(f"☁️ Đang load từ HDFS: {hdfs_path}")
            model = PipelineModel.load(hdfs_path)
            return model, "HDFS Cluster"
        except Exception as e:
            pass

    raise Exception(f"❌ Không thể load model '{model_name}' từ bất kỳ nguồn nào.")

# --- 6. LOAD AVAILABLE MODELS ---
def load_available_models(path="models"):
    models = []
    if os.path.exists(path):
        for item in os.listdir(path):
            if item.endswith("_meta.json"):
                try:
                    with open(os.path.join(path, item), "r", encoding='utf-8') as f:
                        models.append(json.load(f))
                except: continue
    return models