import os
import queue
import json
from langchain.agents import Tool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper, BashProcess
from langchain.agents import initialize_agent, load_tools

from langchain.prompts.chat import (
    SystemMessagePromptTemplate,
)

os.environ["LANGCHAIN_HANDLER"] = "langchain"


def create_agent(agent_type, task):
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
    agent_chain = initialize_agent(
        tools,
        llm,
        agent=agent_type,
        verbose=True,
        memory=memory,
        agent_kwargs={"system_message": f"You are an agent who must do the following: {task}."})

    return agent_chain


def process_task_queue(task_queue, task_executor_agent, task_creation_agent):
    while not task_queue.empty():
        task = task_queue.get()
        print(f"Executing task: {task}")
        result = task_executor_agent.run(input=task)
        print(result)

        custom_prompt = f"{task} {result}"
        new_task_info = task_creation_agent.run(input=custom_prompt)
        new_task_info = json.loads(new_task_info)
        if new_task_info["prompt"]:
            print("Previous task wasn't completed successfully. Creating a new task...")
            print(f"New task: {new_task_info['prompt']}")
            task_queue.put(new_task_info["prompt"])


def main():

    task_queue = queue.Queue()
    print('started')
    while True:
        task = input("Please provide a task (or type 'exit' to quit): ")
        if task.lower() == "exit":
            break
        task_queue.put("Define what your first task should be.")
        task_executor_agent = create_agent("chat-conversational-react-description", task)
        task_creation_llm = ChatOpenAI(temperature=0)
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

        task_creation_prompt = f"You are an agent responsible for creating new sub tasks to fulfill the long term " \
                               f"goal: {task}" \
                                f"You will create new tasks to help fulfill that goal." \
                                f"You will get inputs of the format: " \
                                f"{{ A long task description}} " \
                                f"{{ A long result from attempting to complete that task }}" \
                                f'If this doesn\'t appear to complete the original task described above. ' \
                                f'Create a new task by returning the prompt to be given to a language model to' \
                                f' complete the next step in completing the main task.' \
                                f'To do this, return a json object with key prompt and value None if no new tasks ' \
                                f'need to be created OR with the prompt for the next task to be done if new tasks ' \
                                f'do need to be completed.' \
                                f'Example json object if the task was completed: {{"prompt": None}}' \
                                f'Example json object if the task wasn\'t completed: ' \
                                f'{{"prompt": "new prompt to help complete the task}}"'

        task_creation_agent = initialize_agent(
            [],
            task_creation_llm,
            agent="chat-conversational-react-description",
            verbose=True,
            memory=memory,
            agent_kwargs={"system_message": f"You are an agent who must do the following: {task}."}
        )
        process_task_queue(task_queue, task_executor_agent, task_creation_agent)


if __name__ == "__main__":
    main()
