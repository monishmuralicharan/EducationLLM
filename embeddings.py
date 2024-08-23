#pdf plumber
'''
import pdfplumber

# Example of extracting text using pdfplumber
with pdfplumber.open("example.pdf") as pdf:
    full_text = ""
    for page in pdf.pages:
        full_text += page.extract_text()

print(full_text)
'''

# Use file text and BERT model to convert to vector embeddings in pinecone

pc = Pinecone(
    api_key=os.environ.get("PINECONE_API_KEY")
)
