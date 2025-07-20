import requests
import socket

def test_network():
    try:
        # 测试基本网络连接
        response = requests.get("https://www.google.com", timeout=10)
        print("✅ 基本网络连接正常")
        
        # 测试 Hugging Face 连接
        response = requests.get("https://huggingface.co", timeout=10)
        print("✅ Hugging Face 网站可访问")
        
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络连接问题: {e}")
        
    # 测试 DNS 解析
    try:
        socket.gethostbyname("huggingface.co")
        print("✅ DNS 解析正常")
    except socket.gaierror as e:
        print(f"❌ DNS 解析失败: {e}")

if __name__ == "__main__":
    test_network()