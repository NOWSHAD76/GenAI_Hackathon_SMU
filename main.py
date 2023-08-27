from tools.tools import get_tools
from agents.libra_agent import get_agent
import chainlit as cl
from langchain.agents import Tool, AgentExecutor
from langchain.memory import ConversationBufferWindowMemory

@cl.on_chat_start
def main():
    # Instantiate the chain for that user session
    agent = get_agent()
    tools = get_tools()
    memory = ConversationBufferWindowMemory(k=2)

    agent_executor = AgentExecutor.from_agent_and_tools(agent=agent, tools=tools, verbose=True,memory=memory)

    # Store the chain in the user session
    cl.user_session.set("agent_executor", agent_executor)

@cl.on_message
async def main(message: str):
    # Retrieve the chain from the user session
    agent_executor = cl.user_session.get("agent_executor")  # type: AgentExecutor

    # Call the chain asynchronously
    # res = await agent_executor.acall(message, callbacks=[cl.AsyncLangchainCallbackHandler()])
    res = await cl.make_async(agent_executor)(
        message, callbacks=[cl.LangchainCallbackHandler()]
    )

    # Do any post processing here
    # print(res)

    # "res" is a Dict. For this chain, we get the response by reading the "text" key.
    # This varies from chain to chain, you should check which key to read.
    await cl.Message(content=res['output']).send()


