import os
import streamlit as st 
from apikey import apikey
from langchain.llms import OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain, SequentialChain 
from langchain.memory import ConversationBufferMemory
import PyPDF2
import requests

os.environ['OPENAI_API_KEY'] = apikey

# Streamlit setup
st.title('MACHINE LEARNING')
prompt = st.text_input('Plug in your prompt here') 

# Prompt Templates
title_template = PromptTemplate(
    input_variables=['topic'], 
    template='get me a case based about {topic}'
)

script_template = PromptTemplate(
    input_variables=['title', 'pdf_research'], 
    template='get me information based on this title: {title} while leveraging this PDF research: {pdf_research}'
)

# PDF Text Extraction
def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        with open(pdf_path, "rb") as pdf_file:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            for page in pdf_reader.pages:
                text += page.extract_text()
    except Exception as e:
        st.error("Error extracting PDF text:", e)
    return text

# Search and Extract Relevant Content from PDF
def extract_relevant_content(pdf_text, keyword, context_length=500):
    relevant_content = []

    # Split PDF text into paragraphs
    paragraphs = pdf_text.split('\n\n')

    # Search for the keyword and extract context around it
    for paragraph in paragraphs:
        if keyword.lower() in paragraph.lower():
            start = max(paragraph.lower().find(keyword.lower()) - context_length // 2, 0)
            end = min(start + context_length, len(paragraph))
            relevant_content.append(paragraph[start:end])

    return '\n\n'.join(relevant_content)

# LangChain setup
llm = OpenAI(temperature=0.9)
title_memory = ConversationBufferMemory(input_key='topic', memory_key='chat_history')
script_memory = ConversationBufferMemory(input_key='title', memory_key='chat_history')
title_chain = LLMChain(llm=llm, prompt=title_template, verbose=True, output_key='title', memory=title_memory)
script_chain = LLMChain(llm=llm, prompt=script_template, verbose=True, output_key='script', memory=script_memory)

if prompt:
    pdf_url = "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf"
    response = requests.get(pdf_url)
    with open("downloaded_pdf.pdf", "wb") as pdf_file:
        pdf_file.write(response.content)

    pdf_text = extract_text_from_pdf("downloaded_pdf.pdf")

    # Search keyword (e.g., "machine learning") in PDF and extract relevant content
    keyword = prompt # Replace with your specific search term
    print("prompt",keyword)
    relevant_pdf_text = extract_relevant_content(pdf_text, keyword)

    title = title_chain.run(prompt)
    script = script_chain.run(title=title, pdf_research=relevant_pdf_text)

    st.write(title)
    st.write(script)

    with st.expander('Title History'):
        st.info(title_memory.buffer)

    with st.expander('Script History'):
        st.info(script_memory.buffer)

    with st.expander('Relevant PDF Research'):
        st.info(relevant_pdf_text)
