import os
import requests
from dotenv import load_dotenv

# 加载环境变量
load_dotenv('.env')

def verify_new_token():
    token = os.getenv('HF_TOKEN')
    
    if not token:
        print("❌ 请先将新 token 添加到 .env 文件")
        return
    
    print(f"验证 Token: {token[:10]}...{token[-5:]}")
    print(f"Token 长度: {len(token)} 字符")
    
    # 检查长度
    if len(token) != 37:
        print(f"⚠️ Token 长度异常（期望37字符，实际{len(token)}字符）")
        if len(token) < 37:
            print("💡 可能复制不完整，请重新复制")
        return
    
    # 验证 token 有效性
    headers = {"Authorization": f"Bearer {token}"}
    
    try:
        print("\n🔍 测试 1: 验证 token 有效性...")
        response = requests.get(
            "https://huggingface.co/api/whoami",
            headers=headers,
            timeout=10
        )
        
        if response.status_code == 200:
            user_info = response.json()
            print(f"✅ Token 有效！")
            print(f"用户名: {user_info.get('name', 'Unknown')}")
            print(f"邮箱: {user_info.get('email', 'Unknown')}")
        else:
            print(f"❌ Token 验证失败: {response.status_code}")
            print(f"响应: {response.text}")
            return
            
    except requests.exceptions.RequestException as e:
        print(f"❌ 网络错误: {e}")
        return
    
    # 测试推理 API
    try:
        print("\n🔍 测试 2: 推理 API 访问...")
        inference_url = "https://api-inference.huggingface.co/models/microsoft/DialoGPT-medium"
        payload = {"inputs": "Hello!"}
        
        response = requests.post(
            inference_url,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            print("✅ 推理 API 可用")
        elif response.status_code == 503:
            print("⏳ 模型加载中，API 权限正常")
        else:
            print(f"⚠️ 推理 API 状态: {response.status_code}")
            
    except Exception as e:
        print(f"⚠️ 推理 API 测试: {e}")
    
    print("\n🎉 Token 验证完成！")

if __name__ == "__main__":
    verify_new_token()