import requests
import json
import time
from typing import Optional

class OllamaRemoteTester:
    def __init__(self, base_url: str = "http://192.168.100.255:11434"):
        self.base_url = base_url.rstrip('/')
        self.session = requests.Session()
        self.session.timeout = 30
    
    def test_connection(self) -> bool:
        """测试基础连接"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                print(f"✅ 连接成功: {self.base_url}")
                return True
            else:
                print(f"❌ 连接失败: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            print(f"❌ 连接错误: {e}")
            return False
    
    def list_models(self) -> Optional[list]:
        """获取可用模型列表"""
        try:
            response = self.session.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = response.json().get('models', [])
                print(f"📋 可用模型 ({len(models)} 个):")
                for model in models:
                    name = model.get('name', 'Unknown')
                    size = model.get('size', 0) / (1024**3)  # Convert to GB
                    print(f"  - {name} ({size:.1f}GB)")
                return models
            else:
                print(f"❌ 获取模型列表失败: HTTP {response.status_code}")
                return None
        except Exception as e:
            print(f"❌ 获取模型列表错误: {e}")
            return None
    
    def test_chat(self, model: str = "llama2", message: str = "Hello, how are you?") -> bool:
        """测试聊天功能"""
        try:
            payload = {
                "model": model,
                "messages": [
                    {"role": "user", "content": message}
                ],
                "stream": False
            }
            
            print(f"🤖 测试聊天 (模型: {model})")
            print(f"📝 输入: {message}")
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                reply = result.get('message', {}).get('content', 'No response')
                print(f"💬 回复: {reply}")
                print(f"⏱️  响应时间: {end_time - start_time:.2f}秒")
                return True
            else:
                print(f"❌ 聊天失败: HTTP {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 聊天错误: {e}")
            return False
    
    def test_generate(self, model: str = "llama2", prompt: str = "The sky is") -> bool:
        """测试文本生成功能"""
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False
            }
            
            print(f"📝 测试生成 (模型: {model})")
            print(f"🎯 提示: {prompt}")
            
            start_time = time.time()
            response = self.session.post(
                f"{self.base_url}/api/generate",
                json=payload,
                timeout=60
            )
            end_time = time.time()
            
            if response.status_code == 200:
                result = response.json()
                generated_text = result.get('response', 'No response')
                print(f"✨ 生成: {generated_text}")
                print(f"⏱️  响应时间: {end_time - start_time:.2f}秒")
                return True
            else:
                print(f"❌ 生成失败: HTTP {response.status_code}")
                print(f"错误信息: {response.text}")
                return False
                
        except Exception as e:
            print(f"❌ 生成错误: {e}")
            return False
    
    def run_full_test(self, model: str = "llama2"):
        """运行完整测试套件"""
        print(f"🚀 开始测试远程 Ollama: {self.base_url}")
        print("=" * 50)
        
        # 1. 连接测试
        if not self.test_connection():
            print("❌ 基础连接失败，停止测试")
            return
        
        # 2. 模型列表测试
        models = self.list_models()
        if not models:
            print("⚠️  无法获取模型列表")
        
        # 3. 聊天测试
        print("\n" + "-" * 30)
        self.test_chat(model, "你好，请简单介绍一下自己")
        
        # 4. 生成测试
        print("\n" + "-" * 30)
        self.test_generate(model, "人工智能的未来发展趋势是")
        
        print("\n" + "=" * 50)
        print("🎉 测试完成")


def main():
    # 配置远程 Ollama 服务器地址
    remote_urls = [
        "http://192.168.100.213:11434",  # 局域网服务器
    ]
    
    # 测试模型
    test_models = ["llama3.2:3b"]
    
    for url in remote_urls:
        print(f"\n{'='*60}")
        print(f"测试服务器: {url}")
        print(f"{'='*60}")
        
        tester = OllamaRemoteTester(url)
        
        # 先测试连接
        if tester.test_connection():
            # 获取可用模型
            models = tester.list_models()
            if models:
                # 使用第一个可用模型进行测试
                available_model = models[0]['name']
                tester.run_full_test(available_model)
            else:
                # 如果无法获取模型列表，尝试默认模型
                for model in test_models:
                    print(f"\n尝试模型: {model}")
                    if tester.test_chat(model, "Hello"):
                        break
        else:
            print(f"跳过 {url} - 连接失败")


if __name__ == "__main__":
    main()