import os
import streamlit as st
from apikey import apikey
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain
import pickle

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Load preprocessed data
with open("preprocessed_data.pkl", "rb") as f:
    preprocessed_data = pickle.load(f)
texts = preprocessed_data["texts"]
docsearch = preprocessed_data["docsearch"]
print("this is the main  text", texts)
print("this is the document to search",docsearch)

# ... (rest of your code remains the same)
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Streamlit setup
st.title('project about  OCR')
prompt = st.text_input('Plug in your prompt here')
title_template = PromptTemplate(
    input_variables=['topic', 'pdf_research'],  
    template='get me information based on this title:{topic} while leveraging this PDF research: {pdf_research}'
)

script_template = PromptTemplate(
    input_variables=['pdf_research'], 
    template='{pdf_research}'
)


# LangChain setup
llm = OpenAI(temperature=0.9, max_tokens=300)  # Limiting to 300 tokens
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)


# User input processing
if prompt:
    query = prompt
    docs = docsearch.similarity_search(query)
    qa_result = chain.run(input_documents=docs, question=query)
    # Generate title and script
    title = title_chain.run(topic=prompt, pdf_research=qa_result,)
    script = script_chain.run(title=title, pdf_research=qa_result)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('QA Result'):
        st.info(qa_result)
