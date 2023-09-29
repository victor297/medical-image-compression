# Preprocessing script (run once)
from PyPDF2 import PdfReader
from apikey import apikey
import os
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
import pickle
os.environ['OPENAI_API_KEY'] = apikey

reader = PdfReader('./v1.pdf')
raw_text = ''
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    if text:
        raw_text += text

text_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
)
texts = text_splitter.split_text(raw_text)

embeddings = OpenAIEmbeddings()
docsearch = FAISS.from_texts(texts, embeddings)

# Save preprocessed data
preprocessed_data = {
    "texts": texts,
    "docsearch": docsearch,
}
with open("preprocessed_data.pkl", "wb") as f:
    pickle.dump(preprocessed_data, f)
