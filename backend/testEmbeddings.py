import pinecone


pinecone.init(api_key="PINECONE_API_KEY")  
index_name = "pdf-embeddings"
index = pinecone.Index(index_name)


response = index.fetch(ids=["Homework2-sol"])
print(response)


'''
To test first run embeddings with the pdf of CS251 Homwork sol as sample pdf, then run this to print that shi

'''
