from dotenv import load_dotenv
import os

from transformers import RobertaModel, RobertaTokenizer
import torch
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec
from groq import Groq

# loads environment variables
load_dotenv()

class RAGPipeline():
    def __init__(self):
        # initializes groq and pinecone clients 
        self.client = Groq(api_key=os.environ.get("GROQ_API_KEY"))
        self.pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

        # grabbing the index
        self.index_name = "sample-movies"
        self.index = self.pc.Index(self.index_name)

        # initializing the embedding model
        self.tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
        self.model = RobertaModel.from_pretrained('roberta-large')

    # function to generate embedding from input query
    def text_to_embedding(self, text):
        # Tokenize the input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=128)

        # Generate the embedding by passing the input through the RoBERTa model
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Use the [CLS] token's output as the embedding (first token of the sequence)
        embedding = outputs.last_hidden_state[:, 0, :].squeeze()

        return embedding.numpy()

    def retrieve_context(self, query_embedding, top_k=5):
        # querying the vectorDB
        response = self.index.query(
            vector= query_embedding,
            top_k=top_k,
            include_metadata=True 
        )
        # extracting the context from DB response
        # extracting the context from DB response
        context = ""
        for item in response["matches"]:
            title = str(item['metadata']['title'])
            genre = str(item['metadata']['genre'])
            box_office = float(item['metadata']['box-office'])  # Ensure it's a Python float
            summary = str(item['metadata']['summary'])

            context += f"Title: {title}, Genre: {genre}, Box Office: {box_office}, Summary: {summary}\n"
        
        return context

    def generate_response(self, user_query, context):
        # calling groq model
        context = str(context)
        print("Context:", context)
        chat_completion = self.client.chat.completions.create(
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant providing movie recommendations."
                },
                {
                    "role": "user",
                    "content": user_query,
                },
                {
                    "role":"system",
                    "content": context,
                }
            ],
            model="llama3-8b-8192",
        )
        return chat_completion.choices[0].message.content

    def run_pipeline(self, user_query, state_context):
        embedded_query = self.text_to_embedding(user_query)
        context = self.retrieve_context(embedded_query) + " " + state_context
        response = self.generate_response(user_query, context)
        return response



rag_pipeline = RAGPipeline()


# Type a query, LLM will have all previous context, to end the conversation enter an empty query
state_context = ""
user_query = "Buffer"
while user_query:
    user_query = str(input("Ask a question: "))
    state_context += " " + user_query
    response = rag_pipeline.run_pipeline(user_query, state_context)
    state_context += " " + response
    print(response)
