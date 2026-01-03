import os
import re
import subprocess
import socket

def get_windows_ip():
    try:
        cmd = "chcp 437 & ipconfig"
        output = subprocess.check_output(cmd, shell=True).decode("utf-8", errors="ignore")
        adapter_pattern = re.compile(r"Ethernet adapter vEthernet \(WSL.*?\):(.*?)(Default Gateway|Ethernet adapter|Wireless LAN adapter)", re.DOTALL)
        match = adapter_pattern.search(output)
        if match:
            ip_pattern = re.compile(r"IPv4 Address[ .]*: ([0-9.]+)")
            ip_match = ip_pattern.search(match.group(1))
            if ip_match: return ip_match.group(1)
        return socket.gethostbyname(socket.gethostname())
    except: return "127.0.0.1"

def get_wsl_ip():
    try:
        result = subprocess.check_output(["wsl", "hostname", "-I"]).decode("utf-8").strip()
        return result.split(" ")[0]
    except: return "127.0.0.1"

if __name__ == "__main__":
    print("-" * 30)
    print(f"DETECTED WIN IP: {get_windows_ip()}")
    print(f"DETECTED WSL IP: {get_wsl_ip()}")
    print(f"Master: spark-class org.apache.spark.deploy.master.Master --host {get_windows_ip()} --port 7077")
    print(f"Worker: spark-class org.apache.spark.deploy.worker.Worker spark://{get_windows_ip()}:7077")
    print(f"Start HDFS: start-dfs.sh")
    print(f"HDFS URL: http://localhost:9870/")
    print("-" * 30)