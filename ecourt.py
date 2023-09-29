import os
import streamlit as st 
from apikey import apikey
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
# from langchain.utilities import WikipediaAPIWrapper 
from langchain.tools import Tool
from langchain.utilities import GoogleSearchAPIWrapper
from langchain.document_loaders import WebBaseLoader

url ='https://judgments.ecourts.gov.in//judgments_lib/tmp/2ba19a90778746df96639b605731adedaea550c3daa31fc1730458d114e9ea0c1690287378.pdf'



os.environ['OPENAI_API_KEY'] = apikey
os.environ["GOOGLE_CSE_ID"] = "e106abb4f8c6b4828"
os.environ["GOOGLE_API_KEY"] = "AIzaSyDWEc9ACclimeJ7eER44x97djM68CboX_c"
st.title('E-COURT SEARCH')

prompt = st.text_input('Plug in your prompt here') 

title_template = PromptTemplate(
    input_variables = ['topic'], 
    template=' using https://judgments.ecourts.gov.in/pdfsearch/, search a case about {topic} for me'
)

script_template = PromptTemplate(
    input_variables = ['title','google_research'], 
    template='get me a case about: {title} in india while leveraging on this google reserch:{google_research} '
)

title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')


llm = OpenAI(temperature=0.9) 
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

# wiki = WikipediaAPIWrapper()
search = GoogleSearchAPIWrapper()

# tool = Tool(
#     name="Google Search",
#     description="Search Google for recent results.",
#     func=search.run,
# )
if prompt: 
    title = title_chain.run(prompt)
    g_research = search.run(prompt) 
    
    script = script_chain.run(title=title, google_research=g_research)

    st.write(title) 
    st.write(script) 

    with st.expander('Title History'): 
        st.info(title_memory.buffer)

    with st.expander('Script History'): 
        st.info(script_memory.buffer)

    with st.expander('google Research'): 
        st.info(g_research)