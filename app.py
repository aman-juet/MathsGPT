import streamlit as st 
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents.agent_types import AgentType
from langchain.agents import Tool,initialize_agent
from langchain.callbacks import StreamlitCallbackHandler
from dotenv import load_dotenv
load_dotenv()

#Setting Up streamlit app
st.set_page_config(page_title="Maths GPT",page_icon=":writing_hand:")
st.title("Maths GPT : Text 2 Math Problem Solver and Data Search Assistant")

groq_api_key = st.sidebar.text_input(label="GROQ API KEY",type="password")

if not groq_api_key:
    st.info("Please Provide your GROQ API key to continue")
    st.stop()
    
llm = ChatGroq(model="Gemma2-9b-It",groq_api_key=groq_api_key)

## TOOLS

wiki_wrapper = WikipediaAPIWrapper()
wiki_tool = Tool(
    name="Wikipedia",
    func=wiki_wrapper.run,
    description="A tool for exploring the web for gathering information about the topics mentioned"
)

# MATH TOOl
math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="A tool for answering math related Questions. Only mathematical expressions needs to be provided"
)

prompt = """
You are a agent tasked for solving user mathematical questions. Logically arrive at the solution, provide a detailed expression in bullet points
for the question below
Question : {Question}
Answer:

"""
prompt_template = PromptTemplate(
    input_variables=["Question"],
    template=prompt
)

# Combine all the tool into chain
chain = LLMChain(llm=llm,prompt=prompt_template)

reasoning_tool = Tool(
    name = "Reasoning tool",
    func=chain.run,
    description="A tool for answering logic-based an reasoning questions."
)

#initialize the agents
assistant_agent = initialize_agent(
    tools=[wiki_tool,calculator,reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors = True
)

if "messages" not in st.session_state:
    st.session_state["messages"]=[
        {"role":"assistant","content":"Hi, I am a math chat-bot who can solve all your math queries"}
    ]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg['content'])

# ## func to generate the response
# def gen_response(question):
#     response = assistant_agent.invoke({'input':question})
#     return response


# start the interactions
question = st.text_area("Ask your Questions:")

if st.button("Find my answer"):
    if question:
        with st.spinner("Generate Response.."):
            st.session_state.messages.append({"role":"user","content":question})
            st.chat_message("user").write(question)
            st_cb = StreamlitCallbackHandler(st.container(),expand_new_thoughts=False)
            response = assistant_agent.run(st.session_state.messages,callbacks=[st_cb])
            st.session_state.messages.append({'role':'assistant',"content":response})
            st.write('### Response : ')
            st.success(response)

else: st.warning("Please enter your question") 