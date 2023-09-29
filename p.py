import requests
from PyPDF2 import PdfReader, PdfWriter
import io
import streamlit as st

# Step 1: Download and Combine PDFs
def download_and_combine_pdf(pdf_url):
    response = requests.get(pdf_url)
    pdf_reader = PdfReader(io.BytesIO(response.content))

    return pdf_reader

def combine_pdfs(pdf_readers):
    pdf_writer = PdfWriter()

    for pdf_reader in pdf_readers:
        for page in pdf_reader.pages:
            pdf_writer.add_page(page)

    return pdf_writer

# Combine the provided PDF link
pdf_url = "https://cdnbbsr.s3waas.gov.in/s380537a945c7aaa788ccfcdf1b99b5d8f/uploads/2023/05/2023050195.pdf"
pdf_reader = download_and_combine_pdf(pdf_url)
combined_pdf_writer = combine_pdfs([pdf_reader])

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

def langchain_search_with_context(text, query):
    # Your Langchain/OpenAI search function implementation with context
    # For now, let's do a simple case-insensitive string search

    # Convert the text and query to lowercase for case-insensitive search
    text_lower = text.lower()
    query_lower = query.lower()

    search_results = []
    lines = text_lower.split("\n")

    for i, line in enumerate(lines):
        if query_lower in line:
            start_line = max(0, i - 3)
            end_line = min(len(lines), i + 4)
            context_lines = lines[start_line:end_line]
            result = "\n".join(context_lines)
            search_results.append(result)

    return search_results

# Step 4: Set Up Streamlit App
st.set_page_config(layout="wide")
st.title("PDF Search")
search_query = st.text_input("Enter your search query:")
if st.button("Search"):
    if search_query:
        search_results = langchain_search_with_context(combined_pdf_text, search_query)
        st.header("Search Results")
        if search_results:
            for result in search_results:
                st.write(result)
        else:
            st.write("No results found.")

# Step 5: Run the Streamlit app

st.write("Combined PDF")
st.write(pdf_url)
st.write("Open the Streamlit app to search through the combined PDF.")
