from dotenv import load_dotenv
load_dotenv()

from deepagents import create_deep_agent
from deepagents.backends import FilesystemBackend, LocalShellBackend
from deepagents.backends.local_shell import LocalShellBackend
from langchain.chat_models import init_chat_model
from tools.network import crawl_page


model = init_chat_model("anthropic:MiniMax-M2.5")
agent = create_deep_agent(
    model=model,
    tools=[crawl_page],
    backend=FilesystemBackend(root_dir=".", virtual_mode=True)
)


if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "访问这个页面https://platform.minimaxi.com/docs/api-reference/text-anthropic-api 并做总结，结果写入/results目录"}]})
    
    # Print the agent's response
    for message in result["messages"][-1].content:
        if message["type"]=='text':
            print(message['text'])