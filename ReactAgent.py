from langchain.agents import initialize_agent, Tool
from langchain.tools.python.tool import PythonREPLTool
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler


tools = [
    Tool(
        name="Python",
        func=PythonREPLTool(),
        description="useful for running python. It only outputs the print statements."
    ),
]

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
llm = ChatOpenAI(
    streaming=True,
    callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]),
    temperature=0,
    model_name="gpt-4"
)
# llm = OpenAI(temperature=0)
agent_chain = initialize_agent(
    tools,
    llm,
    agent="chat-conversational-react-description",
    verbose=True,
    memory=memory,
    # return_intermediate_steps=True
)

with open('problem.txt', 'r') as f:
    problem = f.read()

prompt = "Write code to solve the given problem." \
         "Then test the code you have written with python.\n" \
         "Fix any bugs.\n" \
         "Include the corrected python code in your answer, but do not put the final python code in a markdown block\n" \
         f"{problem}\n"

# run the agent
agent_chain.run(prompt)
# response = agent_chain({"input": prompt})
# print(response)