import requests
from bs4 import BeautifulSoup
from PyPDF2 import PdfReader, PdfWriter
import io
import streamlit as st

# Step 1: Download and Combine PDFs
def download_and_combine_pdf(pdf_url):
    try:
        response = requests.get(pdf_url)
        response.raise_for_status()
        pdf_reader = PdfReader(io.BytesIO(response.content))
        return pdf_reader
    except Exception as e:
        st.write(f"Error downloading PDF from {pdf_url}: {e}")
        return None

def combine_pdfs_from_urls(pdf_urls, pdf_writer):
    for url in pdf_urls:
        pdf_reader = download_and_combine_pdf(url)
        if pdf_reader is not None:
            for page in pdf_reader.pages:
                pdf_writer.add_page(page)

def scrape_and_combine_website(url):
    response = requests.get(url)
    soup = BeautifulSoup(response.content, 'html.parser')

    pdf_links = soup.select('a[href$=".pdf"]')
    pdf_urls = [link['href'] for link in pdf_links]

    next_page_link = soup.select_one('a.next')
    if next_page_link:
        next_page_url = next_page_link['href']
        next_page_pdf_urls = scrape_and_combine_website(next_page_url)
        pdf_urls.extend(next_page_pdf_urls)

    return pdf_urls

# Scrape the website and get all PDF URLs, including from nested pages
website_url = "https://legislative.gov.in/constitution-of-india/"
all_pdf_urls = scrape_and_combine_website(website_url)

# Combine all PDFs into one PDF writer
combined_pdf_writer = PdfWriter()
combine_pdfs_from_urls(all_pdf_urls, combined_pdf_writer)

# Save the combined PDF to a file
combined_pdf_filename = "combined.pdf"
with open(combined_pdf_filename, "wb") as output_file:
    combined_pdf_writer.write(output_file)

# Step 2: Extract Text from Combined PDF
def extract_text_from_pdf(pdf_filename):
    with open(pdf_filename, 'rb') as pdf_file:
        pdf_reader = PdfReader(pdf_file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

combined_pdf_text = extract_text_from_pdf(combined_pdf_filename)

# Step 3: Implement Langchain/OpenAI Search (To be added separately, depending on your implementation)
def langchain_search(text, query):
    # Your Langchain/OpenAI search function implementation
    # For now, let's do a simple case-insensitive string search
    return [line for line in text.split("\n") if query.lower() in line.lower()]

# Step 4: Set Up Streamlit App
st.set_page_config(layout="wide")
st.title("PDF Search using Langchain/OpenAI")
search_query = st.text_input("Enter your search query:")
if st.button("Search"):
    if search_query:
        search_results = langchain_search(combined_pdf_text, search_query)
        st.header("Search Results")
        if search_results:
            for result in search_results:
                st.write(result)
        else:
            st.write("No results found.")

# Step 5: Run the Streamlit app
st.write("Combined PDF")
st.write(website_url)
st.write("Open the Streamlit app to search through the combined PDF.")
