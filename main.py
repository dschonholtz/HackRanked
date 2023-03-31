"""
This is a script that uses GPT-4 to solve interview problems.

"""
import openai
import os

# load the openai api key from an env variable
openai.api_key = os.environ["OPENAI_API_KEY"]

from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, LLMChain
from langchain.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    AIMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from langchain.memory import ConversationBufferWindowMemory


def main():
    chat = ChatOpenAI(temperature=0.5, model_name="gpt-4")

    with open("problem.txt", "r") as f:
        problem = f.read()

    template = "You are an extremely effective software engineer doing interview problems."
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{text}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages([system_message_prompt, human_message_prompt])

    chain = LLMChain(llm=chat, prompt=chat_prompt)
    output = chain.run(f"Write the steps you would take to solve this problem. "
                       f"Do not write any code. {problem}")
    print(f"Steps: {output}")
    output = chain.run(f"{problem}"
                       f"{output}"
                       f"Write the code you would write to solve this problem. "
                       f"Do not write any steps.")
    print(output)
    # do runtime analysis
    output = chain.run(f"What is the time and space complexity of this code?"
                       f"{output}")
    print(output)


if __name__ == '__main__':
    main()
