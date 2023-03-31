import os
import subprocess
import queue
import json
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper, BashProcess
from langchain.agents import initialize_agent, load_tools

os.environ["LANGCHAIN_HANDLER"] = "langchain"


def create_agent(agent_type):
    search = SerpAPIWrapper()
    bash = BashProcess()
    llm = ChatOpenAI(temperature=0)
    math_llm = OpenAI(temperature=0.0)

    tools = []

    if agent_type == "chat-conversational-react-description":
        tools = load_tools(["human", "llm-math"], llm=math_llm)

        tools.extend([
            Tool(
                name="Current Search",
                func=search.run,
                description="useful for when you need to answer questions about current events or the current state of the world. the input to this should be a single search term."
            ),
            Tool(
                name="Execute Shell Command",
                func=bash.run,
                description="Executes a shell command and returns the output."
            ),
        ])

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    agent_chain = initialize_agent(tools, llm, agent=agent_type, verbose=True, memory=memory)

    return agent_chain


def process_task_queue(task_queue, task_executor_agent, task_creation_agent):
    while not task_queue.empty():
        task = task_queue.get()
        print(f"Executing task: {task}")
        result = task_executor_agent.run(input=task)
        print(result)

        custom_prompt = f"Parse the output of the execution agent which had prompt: <Start>{task}<end>" \
                        f"\nIt generated the following <start>{result}<end>\n" \
                        f"If the output is correct, return a json object with the key 'prompt' " \
                        f"and the value 'None'.\n" \
                        f"If the output is incorrect or the task isn't completed in it's entirety, " \
                        f"return a json object with the key 'prompt' and the value of the" \
                        f" new prompt which should help finish the task."
        new_task_info = task_creation_agent.run(input=custom_prompt)
        new_task_info = json.loads(new_task_info)
        if new_task_info["prompt"]:
            print("Previous task wasn't completed successfully. Creating a new task...")
            print(f"New task: {new_task_info['prompt']}")
            task_queue.put(new_task_info["prompt"])


def main():
    task_executor_agent = create_agent("chat-conversational-react-description")
    task_creation_agent = ChatOpenAI(temperature=0)

    task_queue = queue.Queue()

    while True:
        task = input("Please provide a task (or type 'exit' to quit): ")
        if task.lower() == "exit":
            break

        task_queue.put(task)
        process_task_queue(task_queue, task_executor_agent, task_creation_agent)


if __name__ == "__main__":
    main()
