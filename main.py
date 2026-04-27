from dotenv import load_dotenv
load_dotenv()

import os
from pathlib import Path
from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, CompositeBackend
from langchain.chat_models import init_chat_model
from tools.network import crawl_page
from tools.audio import text_to_speech
from backends.redis_backend import RedisConfig, RedisBackend

from langfuse import get_client
from langfuse.langchain import CallbackHandler

langfuse = get_client()

if langfuse.auth_check():
    print("Langfuse client is authenticated and ready!")
else:
    print("Authentication failed. Please check your credentials and host.")
langfuse_handler = CallbackHandler()


WORK_DIR = Path(__file__).parent
SYSTEM_PROMPT = """你是一个阅读者，我将给一些博客或咨询网页的地址\
你使用工具来浏览这个地址，并获得其内容，然后将内容进行总结，不超过500字。如果内容段可以适当减少最后的总结的字数。总结的内容保存到/results/目录下\
之后你可以使用voice_agent(一个subagent）将总结的内容从/rusults/中读出并转换成音频，最后保存本地。"""
redis_url=os.getenv("REDIS_DB_URL","")
r_backend = RedisBackend(RedisConfig(redis_url))


voice_agent = {
    "name": "voice_agent",
    "description": "一个用来将文本转成语音mp3的助手",
    "system_prompt": "你是一个将文字转换成音频的专家，你到/results/中读取已经总结好的文本，并利用工具将它们转化成MP3音频，并将结果保存至/results/目录下。\
    你需要返回音频的存储地址，但严禁返回音频的内容（包括编码后的内容）",
    "tools": [text_to_speech],
    # "backend":FilesystemBackend(root_dir=WORK_DIR, virtual_mode=True),
}


model = init_chat_model("anthropic:MiniMax-M2.5", max_tokens=196608)
agent = create_deep_agent(
    model=model,
    tools=[crawl_page],
    backend=CompositeBackend(
        default=r_backend,
        routes={
            "/results/": FilesystemBackend(root_dir=str(WORK_DIR/"results"), virtual_mode=True)
        }
    ),
    system_prompt=SYSTEM_PROMPT,
    subagents=[voice_agent],
)


if __name__ == "__main__":
    result = agent.invoke(
        {"messages": [
            {"role": "user", "content": "访问这个页面\
                https://docs.langchain.com/oss/python/deepagents/permissions\
                并总结"}
        ]},
        config={"callbacks":[langfuse_handler]}
    )
    
    # Print the agent's response
    for message in result["messages"][-1].content:
        if message["type"]=='text':
            print(message['text'])