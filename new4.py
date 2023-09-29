import os
import streamlit as st
from apikey import apikey
from PyPDF2 import PdfReader
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
from langchain.chains.question_answering import load_qa_chain

# Set the OpenAI API key
os.environ['OPENAI_API_KEY'] = apikey

# Load the PDF and extract text
reader = PdfReader('./v1.pdf')
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

# Split the text into smaller chunks
text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

# Download embeddings from OpenAI
embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)
chain = load_qa_chain(OpenAI(), chain_type="stuff")

# Streamlit setup
st.title('project about  OCR')
prompt = st.text_input('Plug in your prompt here')
title_template = PromptTemplate(
    input_variables=['topic'], 
    template='get me information based on this title {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'pdf_research'], 
    template='get me information based on this title: {title} while leveraging this PDF research: {pdf_research}'
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
    title = title_chain.run(prompt)
    script = script_chain.run(title=title, pdf_research=qa_result)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('QA Result'):
        st.info(qa_result)
