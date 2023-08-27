from tools.tools import get_tools
from utilities.custom_parsers import get_custom_output_parser,get_prompt
from langchain import LLMChain
from langchain.chat_models import ChatOpenAI
from langchain.agents import LLMSingleActionAgent

tools = get_tools()

template = """
You are a library assitant specialized in checking whether a citation is in valid format or not.You have access to the following tools:

{tools}

There is tool human to chat with human in case you need any clarification.

Strictly use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer (After this always provide the next step)
Final Answer: the final answer to the original input question once you are done with all actions and have an answer

Begin! Remember that you are a librarian and for any type of question related to citation you should provide all the components of the citation.

Previous conversation history:
{history}

Question: {input}
{agent_scratchpad}"""

prompt = get_prompt(template)

# Initiate our LLM - default is 'gpt-3.5-turbo'
llm = ChatOpenAI(temperature=0,openai_api_key="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")

# LLM chain consisting of the LLM and a prompt
llm_chain = LLMChain(llm=llm, prompt=prompt)

# Using tools, the LLM chain and output_parser to make an agent
tool_names = [tool.name for tool in tools]

output_parser = get_custom_output_parser()

def get_agent():
    agent = LLMSingleActionAgent(
        llm_chain=llm_chain,
        output_parser=output_parser,
        # We use "Observation" as our stop sequence so it will stop when it receives Tool output
        # If you change your prompt template you'll need to adjust this as well
        stop=["\nObservation:"],
        allowed_tools=tool_names
    )
    return agent