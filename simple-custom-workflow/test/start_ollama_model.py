import requests
import json
import time
import subprocess
import sys

def check_ollama_service():
    """检查Ollama服务是否运行"""
    try:
        response = requests.get("http://192.168.100.213:11434/api/tags", timeout=5)
        return response.status_code == 200
    except:
        return False

def load_model(model_name):
    """加载指定模型"""
    print(f"加载模型: {model_name}")
    
    try:
        # 发送一个简单请求来加载模型
        response = requests.post(
            "http://192.168.100.213:11434/api/generate",
            json={
                "model": model_name,
                "prompt": "Hello",
                "stream": False
            },
            timeout=60
        )
        
        if response.status_code == 200:
            print(f"✓ 模型 {model_name} 加载成功")
            return True
        else:
            print(f"✗ 模型 {model_name} 加载失败，状态码: {response.status_code}")
            return False
    except Exception as e:
        print(f"加载模型失败: {e}")
        return False

def check_model_status():
    """检查模型状态"""
    try:
        # 检查运行中的模型
        response = requests.get("http://192.168.100.213:11434/api/ps", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            if models:
                print("\n当前运行的模型:")
                for model in models:
                    print(f"  - {model['name']}")
                    print(f"    大小: {model.get('size', 'Unknown')}")
                    print(f"    到期时间: {model.get('expires_at', 'Unknown')}")
            else:
                print("当前没有运行中的模型")
        
        # 检查已安装的模型
        response = requests.get("http://192.168.100.213:11434/api/tags", timeout=5)
        if response.status_code == 200:
            models = response.json().get('models', [])
            print(f"\n已安装的模型数量: {len(models)}")
            for model in models:
                print(f"  - {model['name']}")
    except Exception as e:
        print(f"检查模型状态失败: {e}")

def main():
    print("=== Ollama模型启动脚本 ===")
    
    # 检查服务状态
    if not check_ollama_service():
        if not start_ollama_service():
            print("无法启动Ollama服务，请检查安装")
            sys.exit(1)
    else:
        print("✓ Ollama服务已运行")
    
    # 要启动的模型列表
    models_to_load = ["llama3.2:3b"]
    
    # 加载模型
    for model in models_to_load:
        load_model(model)
        time.sleep(2)  # 等待模型加载
    
    # 检查最终状态
    print("\n=== 最终状态 ===")
    check_model_status()

if __name__ == "__main__":
    main()