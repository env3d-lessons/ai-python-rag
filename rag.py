from sentence_transformers import SentenceTransformer
import faiss


def search(query, top_k=3):
    query_emb = model.encode([query])
    D, I = index.search(query_emb, top_k)
    print(f"\nQuery: {query}\n")
    print(I)
    for i, idx in enumerate(I[0]):
        verse_ref, verse_text = verses[idx].split('\t')
        print(f"Rank {i+1} | {verse_ref}: {verse_text}")
        print(f"  (Distance: {D[0][i]:.4f})\n")

verses = []

f = open('kjv.txt', encoding='utf-8-sig')
for line in f:
    verses.append(line.strip())    
    if 'Genesis' not in line:
        break

f.close()

model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
embeddings = model.encode(verses)

# Create FAISS index
index = faiss.IndexFlatL2(384)
index.add(embeddings)
