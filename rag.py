from sentence_transformers import SentenceTransformer
import faiss
import os

# Load the SentenceTransformer model
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

verses = []

f = open('kjv.txt', encoding='utf-8-sig')
verses = f.readlines()
f.close()


def search(query, top_k=3):
    index = load_index()
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)
    return [ verses[i] for i in I[0]]


def load_index(index_file='faiss_index.bin', embedding_dim=384):
    # Check if the FAISS index file exists
    if os.path.exists(index_file):
        # Load the FAISS index from the file
        index = faiss.read_index(index_file)
    else:
        # Read the verses from the file
        with open('kjv.txt', encoding='utf-8-sig') as f:
            verses = f.readlines()

        # Encode the verses
        embeddings = model.encode(verses)

        # Create FAISS index
        index = faiss.IndexFlatL2(embedding_dim)
        index.add(embeddings)

        # Save the FAISS index to a file
        faiss.write_index(index, index_file)

    return index
