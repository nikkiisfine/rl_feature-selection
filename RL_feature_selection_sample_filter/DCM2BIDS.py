import os
import subprocess
import re
from pathlib import Path

# --- 路径配置 ---
BASE_DIR = "/Users/daxiaoniqiu/anaconda3/AI4H/pythonProject/myvenv/thesis/thesis"
DICOM_ROOT = os.path.join(BASE_DIR, "AD_main_experiment/ADNI")
CONFIG_FILE = os.path.join(BASE_DIR, "AD_MVP/dcm2bids_config.json")
BIDS_OUTPUT = os.path.join(BASE_DIR, "AD_main_experiment/BIDS")

def extract_session_id(path_string):
    """
    从路径中提取日期格式 (YYYY-MM-DD)，并转换为 BIDS 合规格式 (YYYYMMDD)
    例如: /2017-04-17_11_22_08.0/ -> 20170417
    """
    # 匹配 YYYY-MM-DD 格式的正则
    match = re.search(r'(\d{4})-(\d{2})-(\d{2})', path_string)
    if match:
        return "".join(match.groups())
    return None

def run_batch_conversion():
    # 确保输出目录存在
    os.makedirs(BIDS_OUTPUT, exist_ok=True)

    # 遍历 ADNI 目录
    # ADNI 结构通常是: ADNI/Subject_ID/Protocol_Name/Date_Time/Image_ID
    # for subj_id in ['002_S_1280']:
    for subj_id in os.listdir(DICOM_ROOT):
        subj_path = os.path.join(DICOM_ROOT, subj_id)
        if not os.path.isdir(subj_path):
            continue

        # 清洗 Subject ID (002_S_6009 -> 002S6009)
        clean_subj_id = subj_id.replace("_", "")
        
        # 深度遍历该受试者的所有文件夹以寻找日期
        found_sessions = {}
        
        for root, dirs, files in os.walk(subj_path):
            # 只有当文件夹内包含 .dcm 文件时才处理
            if any(f.lower().endswith(".dcm") for f in files):
                session_id = extract_session_id(root)
                
                if session_id and session_id not in found_sessions:
                    print(f"找到受试者 {clean_subj_id} 的场次: {session_id}")
                    
                    # 构建 dcm2bids 命令
                    # 注意：-d 参数指向受试者的顶层目录，dcm2bids 会自动递归搜索
                    # 但如果我们想按 session 转换，建议在循环中执行
                    cmd = [
                        "dcm2bids",
                        "-d", subj_path,
                        "-p", clean_subj_id,
                        "-s", session_id,
                        "-c", CONFIG_FILE,
                        "-o", BIDS_OUTPUT,
                        "--force_dcm2bids"
                    ]
                    
                    try:
                        subprocess.run(cmd, check=True)
                        found_sessions[session_id] = True # 防止同一场次重复转换
                    except subprocess.CalledProcessError as e:
                        print(f"转换失败: {clean_subj_id}, Session: {session_id}, 错误: {e}")

if __name__ == "__main__":
    run_batch_conversion()