import os
import subprocess
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.utilities import SerpAPIWrapper, BashProcess
from langchain.agents import initialize_agent

os.environ["LANGCHAIN_HANDLER"] = "langchain"


def create_agent():
    # search = SerpAPIWrapper()
    bash = BashProcess()
    tools = [
        # Tool(
        #     name="Current Search",
        #     func=search.run,
        #     description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
        # ),
        Tool(
            name="Execute Shell Command",
            func=bash.run,
            description="Executes a shell command and returns the output."
        ),
    ]

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    llm = ChatOpenAI(temperature=0)
    agent_chain = initialize_agent(tools, llm, agent="chat-conversational-react-description", verbose=True,
                                   memory=memory)

    return agent_chain


def main():
    agent = create_agent()

    while True:
        task = input("Please provide a task (or type 'exit' to quit): ")
        if task.lower() == "exit":
            break

        result = agent.run(input=task)
        print(result)


if __name__ == "__main__":
    main()
