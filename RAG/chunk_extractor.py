pdf_name = "knowledge.pdf"
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter

def chunker_extractor(file_path):
    # Open the PDF file in binary mode
    pdf_file = open(file_path, 'rb')
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    
    # Extract text from all pages
    text = ""
    for page_num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[page_num]
        text += page.extract_text()
    
    # Close the PDF file
    pdf_file.close()
    
    # Initialize the text splitter
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    
    # Split the text into chunks
    text_chunks = text_splitter.create_documents([text])
    
    # Extract the text content from the Document objects and return as a list of strings
    return text_chunks

