import os
from tavily import TavilyClient
from dotenv import load_dotenv
from pathlib import Path

# 加载 .env（用我们之前验证过的手动解析方式）
env_path = Path("D:/Desktop/agent/.env").resolve()
load_dotenv(dotenv_path=env_path, encoding="utf-8")
tavily_api_key = os.getenv("TAVILY_API_KEY")

print(f"🔑 读取到的 Tavily Key：{tavily_api_key[:10]}...")

# 测试 Tavily 调用
try:
    tavily = TavilyClient(api_key=tavily_api_key)
    # 简单搜索测试
    response = tavily.search(
        query="北京晴天适合去的景点",
        search_depth="basic",  # 免费版只能用 basic
        include_answer=True,
        timeout=20  # 延长超时时间
    )
    print("✅ Tavily API 调用成功！")
    print(f"📌 返回结果：{response.get('answer', '无直接答案')}")
except Exception as e:
    print(f"❌ Tavily API 调用失败：{str(e)[:100]}")
    # 备选方案：打印错误详情
    import traceback
    traceback.print_exc()