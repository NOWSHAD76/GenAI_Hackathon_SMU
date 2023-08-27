from langchain.schema import (
    AIMessage,
    HumanMessage,
    SystemMessage
)
from langchain.agents import tool,load_tools,Tool
from langchain.chat_models import ChatOpenAI
from langchain import PromptTemplate, OpenAI, LLMChain

def get_tools():
    system_message_content = """
    You are a helpful library assistant that specialized in checking whether a citation is in valid format or not and identify all the components of the citation.
    Remember that you just need to answer the question and leave the conversation. Do not wait for any other instructions.
    """

    # @tool
    # def validate_citation(citation: str) -> str:
    #     """Returns the whether the citation is right APA format or not."""
    #     chat = ChatOpenAI(openai_api_key="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",temperature=0)
    #     messages = [
    #         SystemMessage(content=system_message_content),
    #         HumanMessage(content=citation)
    #         ]
    #     res = chat(messages)
    #     return res.content

    @tool
    def validate_citation(citation: str) -> str:
        """Returns the whether the citation is right APA format or not."""
        template = """
        You are a helpful library assistant that specialized in checking whether a citation is in valid format or not and identify all the components of the citation.
        
        Below are some of the samples of the correct format of APA citation for each type of source for your reference. Here "A." refers to first letter from the name of author.
        Book : Author, A. A. (Year of publication). Title of work: Capital letter to start subtitle. Publisher.

        Journal Article : Author, A. A., & Author, B. B. (Date of publication). Title of article: Capital letter to start subtitle. Title of Online Periodical, volume number(issue number if available). https://www.someaddress.com

        Newspaper Article : Author, A. A. (Year, Month Day). Title of article: Capital letter to start subtitle. Title of Newspaper. https://www.someaddress.com

        Magazine Article : Author, A. A. (Year, Month Day). Title of article: Capital letter to start subtitle. Title of Magazine. https://www.someaddress.com

        Website : Author, A. A. (Year, Month Day). Title of webpage: Capital letter to start subtitle. Website Name. https://www.someaddress.com

        Social Media : Author, A. A. [@username]. (Year, Month Day). First 20 words of post [Description of format]. Site Name. https://www.someaddress.com

        Video : Author, A. A. [Screen name]. (Year, Month Day). Title of video: Capital letter to start subtitle [Video]. Site Name. https://www.someaddress.com

        Image : Creator, A. A. (Year). Title of image: Capital letter to start subtitle [Description of format]. Site Name. https://www.someaddress.com

        Personal Communication : Name of person who communicated with you. (Year, Month Day). Description of communication.

        Secondary Source : Author, A. A. (Year of publication). Title of work: Capital letter to start subtitle. Publisher. (Original work published Year)

        Note: Strictly validate the citation based on above samples and no deviations are allowed.

        Given below citation you need to validate and respond whether it is valid or not.
        Follow the below steps exactly to validate the citation.
        1) Get all the components of the citation
        2) Validate the components against the examples provided earlier for reference.
        3) If it is matching any of the examples then it is correct otherwise it is wrong.

        Provide all the components of the citation and explanation if the citation is wrong.
        Citation: {citation}
        """
        prompt = PromptTemplate(template=template, input_variables=["citation"])
        llm_chain = LLMChain(prompt=prompt, llm=OpenAI(temperature=0,openai_api_key="insert_the_key"), verbose=False)
        res = llm_chain.run(citation)
        return res

    human = load_tools(["human"])

    tools = [
            Tool(
                name="Citation validater",
                func=validate_citation,
                description="useful when you want to validate whether the citation is in APA format or not",
            )
            #,
            # Tool(
            #     name=human[0].name,
            #     func=human[0].input_func,
            #     description=human[0].description,
            # )
        ]

    return tools
