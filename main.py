from deepagents import create_deep_agent
from langchain.chat_models import init_chat_model
from dotenv import load_dotenv

load_dotenv()

model = init_chat_model("anthropic:MiniMax-M2.5")
agent = create_deep_agent(
    model=model,
)


if __name__ == "__main__":
    result = agent.invoke({"messages": [{"role": "user", "content": "What is langgraph?"}]})
    
    # Print the agent's response
    print(result["messages"][-1].content)