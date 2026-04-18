# Home Deep Agents

一个基于 [deepagents](https://docs.langchain.com/oss/python/deepagents/) 框架的多 Agent 系统，专注于网页内容抓取、总结与语音合成。

## 功能概述

- **主 Agent**：接收用户提供的网页 URL，使用工具抓取页面内容并进行总结（不超过 500 字）
- **语音子 Agent（voice-agent）**：将总结好的文本内容转换为 MP3 音频，保存到本地
- **网络工具**：集成 Tavily API，支持网页搜索与内容抓取
- **音频工具**：集成 MiniMax T2A API，支持文本转语音

## 项目结构

```
home-deep-agents/
├── main.py              # 主入口，定义 Agent 和 SubAgent
├── requirements.txt     # 依赖列表
├── tools/
│   ├── network.py        # 网络工具（网页搜索 & 抓取）
│   └── audio.py          # 音频工具（文字转语音）
├── results/              # 保存总结文本和生成的 MP3 音频
└── .env                  # 环境变量配置（API Key 等）
```

## 环境配置

在项目根目录创建 `.env` 文件，填入以下环境变量：

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
TAVILY_API_KEY=your_tavily_api_key_here
```

- `ANTHROPIC_API_KEY`：用于 MiniMax TTS API 鉴权（填入 Anthropic 平台的 API Key）
- `TAVILY_API_KEY`：[Tavily](https://tavily.com) 平台 API Key，用于网页搜索和内容抓取

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行

```bash
python main.py
```

主 Agent 会抓取指定网页并总结，语音子 Agent 会将总结内容转换为 MP3 音频保存到 `results/` 目录。

## 工具说明

### 网络工具（tools/network.py）

| 工具 | 功能 |
|------|------|
| `internet_search` | 使用 Tavily 进行网络搜索 |
| `crawl_page` | 抓取指定 URL 的页面内容 |

### 音频工具（tools/audio.py）

| 工具 | 功能 |
|------|------|
| `text_to_speech` | 将文本转换为 MP3 音频，支持语速、音调、情感等参数调节 |

## 主要依赖

- **deepagents**：多 Agent 框架
- **langchain / langchain-anthropic**：Agent 构建与 LLM 调用
- **tavily-python**：网页搜索与抓取
- **httpx**：HTTP 客户端
- **requests**：TTS API 调用
