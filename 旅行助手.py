import re
import os
import time
import hmac
import hashlib
import requests
from openai import OpenAI
from tavily import TavilyClient
from dotenv import load_dotenv
from pathlib import Path

# ===================== 加载.env配置 =====================
env_path = Path("D:/Desktop/agent/.env").resolve()
load_dotenv(dotenv_path=env_path, encoding="utf-8")

# 读取LLM和Tavily配置
API_KEY = os.getenv("LLM_API_KEY")
BASE_URL = os.getenv("LLM_BASE_URL")
MODEL_ID = os.getenv("LLM_MODEL_ID")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
AMAP_API_KEY = os.getenv("AMAP_API_KEY")

# ===================== 系统提示词 =====================
AGENT_SYSTEM_PROMPT = """
你是一个智能旅行助手。你的任务是分析用户的请求，并使用可用工具一步步地解决问题。

# 可用工具:
- `get_weather(city: str)`: 查询指定城市的实时天气。
- `get_attraction(city: str, weather: str)`: 根据城市和天气搜索推荐的旅游景点。

# 输出格式要求:
你的每次回复必须严格遵循以下格式，包含一对Thought和Action：

Thought: [你的思考过程和下一步计划]
Action: [你要执行的具体行动]

Action的格式必须是以下之一：
1. 调用工具：function_name(arg_name="arg_value")
2. 结束任务：Finish[最终答案]

# 重要提示:
- 每次只输出一对Thought-Action
- Action必须在同一行，不要换行
- 当收集到足够信息可以回答用户问题时，必须使用 Action: Finish[最终答案] 格式结束

请开始吧！
"""

# ===================== 工具函数 =====================
def get_weather(city: str) -> str:
    """
    查询指定城市的实时天气，返回可读的天气描述
    """

    # 第一步：获取城市 adcode
    geo_url = "https://restapi.amap.com/v3/geocode/geo"
    geo_params = {
        "key": AMAP_API_KEY,
        "address": city,
        "output": "json"
    }

    try:
        geo_res = requests.get(geo_url, params=geo_params, timeout=10).json()
        if geo_res.get("status") != "1":
            return f"无法识别城市：{city}"

        adcode = geo_res["geocodes"][0]["adcode"]

        # 第二步：查询实时天气
        weather_url = "https://restapi.amap.com/v3/weather/weatherInfo"
        weather_params = {
            "key": AMAP_API_KEY,
            "city": adcode,
            "extensions": "base",
            "output": "json"
        }

        weather_res = requests.get(weather_url, params=weather_params, timeout=10).json()
        if weather_res.get("status") != "1":
            return f"天气查询失败：{weather_res.get('info', '未知错误')}"

        # 解析并返回天气信息
        live = weather_res["lives"][0]
        return (f"{live['city']}当前天气：{live['weather']}，"
                f"气温 {live['temperature']}℃，{live['winddirection']}风 {live['windpower']}级")

    except Exception as e:
        return f"天气查询异常：{str(e)[:50]}"

# 测试景点推荐接口
def get_attraction(city: str, weather: str) -> str:
    """
    根据城市和天气推荐景点，返回可读的景点列表
    """
    if not AMAP_API_KEY:
        return "未配置高德地图 API Key"

    # 第一步：获取城市 adcode
    geo_url = "https://restapi.amap.com/v3/geocode/geo"
    geo_params = {
        "key": AMAP_API_KEY,
        "address": city,
        "output": "json"
    }

    try:
        geo_res = requests.get(geo_url, params=geo_params, timeout=10).json()
        if geo_res.get("status") != "1":
            return f"无法识别城市：{city}"

        adcode = geo_res["geocodes"][0]["adcode"]

        # 第二步：根据天气选择 POI 类型
        if "雨" in weather:
            types = "110000|120000|140000"  # 室内：博物馆、展览馆、商场
            type_desc = "室内景点（博物馆、商场等）"
        else:
            types = "110100|110101|110102|110103"  # 户外：公园、风景名胜、古迹
            type_desc = "户外景点（公园、风景名胜等）"

        # 第三步：POI 搜索
        poi_url = "https://restapi.amap.com/v3/place/text"
        poi_params = {
            "key": AMAP_API_KEY,
            "keywords": "景点",
            "types": types,
            "city": adcode,
            "citylimit": True,
            "output": "json",
            "page_size": 5
        }

        poi_res = requests.get(poi_url, params=poi_params, timeout=10).json()
        if poi_res.get("status") != "1" or not poi_res.get("pois"):
            return f"{city}暂无推荐景点，可尝试其他城市或天气条件。"

        # 格式化景点列表
        attractions = []
        for poi in poi_res["pois"][:3]:
            attractions.append(f"- {poi.get('name', '未知景点')}：{poi.get('address', '地址未知')}")

        return (f"{city} {weather}天气推荐：\n"
                f"推荐类型：{type_desc}\n"
                + "\n".join(attractions))

    except Exception as e:
        return f"景点推荐异常：{str(e)[:50]}"

# 工具映射
available_tools = {"get_weather": get_weather, "get_attraction": get_attraction}

# ===================== LLM客户端 =====================
class OpenAICompatibleClient:
    def __init__(self, model: str, api_key: str, base_url: str):
        self.model = model
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def generate(self, prompt: str, system_prompt: str) -> str:
        print("正在调用大模型...")
        try:
            messages = [
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': prompt}
            ]
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, temperature=0.7, stream=False
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"LLM调用错误: {e}")
            return "错误:调用语言模型失败"

# ===================== 主程序 =====================
if __name__ == "__main__":
    llm = OpenAICompatibleClient(model=MODEL_ID, api_key=API_KEY, base_url=BASE_URL)
    user_prompt = "帮我查今天北京的天气，然后推荐合适的旅游景点"
    prompt_history = [f"用户请求: {user_prompt}"]

    print(f"用户输入: {user_prompt}\n" + "="*40)

    for i in range(5):
        print(f"--- 循环 {i+1} ---\n")
        full_prompt = "\n".join(prompt_history)
        llm_output = llm.generate(full_prompt, AGENT_SYSTEM_PROMPT)
        
        # 截断多余输出
        match = re.search(r'(Thought:.*?Action:.*?)(?=\n\s*(?:Thought:|Action:|Observation:)|\Z)', llm_output, re.DOTALL)
        if match:
            llm_output = match.group(1).strip()
        print(f"模型输出:\n{llm_output}\n")
        prompt_history.append(llm_output)
        
        # 解析Action
        action_match = re.search(r"Action: (.*)", llm_output, re.DOTALL)
        if not action_match:
            observation = "错误:未解析到Action字段"
        else:
            action_str = action_match.group(1).strip()
            if action_str.startswith("Finish"):
    # 先匹配，再判断是否成功
                finish_match = re.match(r"Finish\[(.*)\]", action_str)
                if finish_match:
                    final_answer = finish_match.group(1)
                    print(f"\n任务完成，最终答案：\n{final_answer}")
                    break
                else:
                    # 匹配失败时，直接把整个 Action 作为答案，避免崩溃
                    final_answer = action_str.replace("Finish", "").strip("[] ")
                    print(f"\n未严格匹配 Finish 格式，直接返回：\n{final_answer}")
                    break   
            try:
                tool_name = re.search(r"(\w+)\(", action_str).group(1)
                args_str = re.search(r"\((.*)\)", action_str).group(1)
                kwargs = dict(re.findall(r'(\w+)="([^"]*)"', args_str))
                observation = available_tools[tool_name](**kwargs) if tool_name in available_tools else f"未知工具: {tool_name}"
            except Exception as e:
                observation = f"工具调用失败: {e}"
        
        observation_str = f"Observation: {observation}"
        print(f"{observation_str}\n" + "="*40)
        prompt_history.append(observation_str)
    else:
        print("警告:达到最大循环次数")