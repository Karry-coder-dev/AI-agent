# 智能旅行助手
基于大语言模型的旅行助手，支持实时天气查询和景点推荐（高德地图API）。

## 快速开始
1. 复制 `.env.example` 为 `.env`，填写自己的 API Key；
2. 安装依赖：`pip install requests python-dotenv openai tavily-python`；
3. 运行：`python 旅行助手.py`。

## 功能
- 调用高德地图API查询实时天气；
- 根据天气推荐室内/户外景点（高德POI搜索）；
- 大模型整合信息，生成完整旅行建议。