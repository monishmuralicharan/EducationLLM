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


import pdfplumber
from transformers import RobertaTokenizer, RobertaModel
import torch
import pinecone
import os

# Initialize Pinecone and RoBERTa
pinecone.init(api_key=os.environ.get("8617f8ef-56a3-44d5-a50c-bc5afe9df4f2"))
index_name = "your-pinecone-index"
index = pinecone.Index(index_name)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')

# Function to extract text from a PDF using pdfplumber
def extract_text_from_pdf(pdf_path):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text()
    return text

# Function to convert text to embeddings
def text_to_embedding(text):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze()
    return embedding.numpy()

# Function to push embeddings to Pinecone
def push_to_pinecone(embedding, metadata):
    index.upsert(vectors=[(metadata['id'], embedding)], metadata=metadata)

# Example usage
pdf_path = "your-document.pdf"
text = extract_text_from_pdf(pdf_path)
embedding = text_to_embedding(text)

# Define metadata for your PDF (e.g., title, author, etc.)
metadata = {
    "id": "unique-id-for-this-document",
    "title": "Document Title",
    "author": "Author Name",
    "category": "Some Category"
}

# Push the embedding and metadata to Pinecone
push_to_pinecone(embedding, metadata)


'''
pc = Pinecone(
   api_key=os.environ.get("PINECONE_API_KEY")
)
'''
