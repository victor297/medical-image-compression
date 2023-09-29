import streamlit as st 
from langchain.document_loaders.sitemap import SitemapLoader
from langchain.document_loaders import WebBaseLoader
url ='https://judgments.ecourts.gov.in'
sitemap_loader = SitemapLoader(web_path=url)

docs = sitemap_loader.load()
st.write(docs)

st.title('E-COURT SEARCH')
loader = WebBaseLoader(url)
scrape_data = loader.load()
st.write(scrape_data)