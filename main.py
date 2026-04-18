from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.local_shell import LocalShellBackend
from langchain.chat_models import init_chat_model
from tools.network import crawl_page
from tools.audio import text_to_speech



WORK_DIR = Path(__file__).parent
SYSTEM_PROMPT = """你是一个阅读者，我将给一些博客或咨询网页的地址\
你使用工具来浏览这个地址，并获得其内容，然后将内容进行总结，不超过500字。如果内容段可以适当减少最后的总结的字数。总结的内容保存到/results中\
之后你可以使用voice_agent(一个subagent）将总结的内容转换成音频，并保存本地。"""


voice_agent = {
    "name": "voice-agent",
    "description": "一个用来将文本转成语音mp3的助手",
    "system_prompt": "你是一个将文字转换成音频的专家，你可以到/results中读取已经总结好的文本，并利用工具将它们转化成MP3音频。",
    "tools": [text_to_speech],
    "backend":FilesystemBackend(root_dir=WORK_DIR, virtual_mode=True),
}


model = init_chat_model("anthropic:MiniMax-M2.5")
agent = create_deep_agent(
    model=model,
    tools=[crawl_page],
    backend=FilesystemBackend(root_dir=WORK_DIR, virtual_mode=True),
    system_prompt=SYSTEM_PROMPT,
    subagents=[voice_agent],
)


if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "访问这个页面https://docs.langchain.com/oss/python/deepagents/subagents并总结"}]})
    
    # Print the agent's response
    for message in result["messages"][-1].content:
        if message["type"]=='text':
            print(message['text'])