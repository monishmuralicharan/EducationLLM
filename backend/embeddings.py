import pypdf
from transformers import RobertaTokenizer, RobertaModel
import torch
import pinecone
import os

# Initialize Pinecone and RoBERTa
pinecone.init(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = "pdf-embeddings"
index = pinecone.Index(index_name)

tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
model = RobertaModel.from_pretrained('roberta-large')

# Function to extract text from a PDF using pypdf
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        reader = pypdf.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
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
pdf_path = "CS251Homework2Solution.pdf"
text = extract_text_from_pdf(pdf_path)
embedding = text_to_embedding(text)

# Define metadata for your PDF (e.g., title, author, etc.)
metadata = {
    "id": "Homework2-sol",
    "title": "CS251Homework2Solution",
    "author": "ANDRES BOSADA",
    "category": "DSA Solution"
}

# Push the embedding and metadata to Pinecone
push_to_pinecone(embedding, metadata)
