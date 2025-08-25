import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from embedding_sources import GoodbooksEmbeddings
from vibe_methods import PCAVibeIdentifier

# test basic loading
embeddings = GoodbooksEmbeddings("data/raw", n_components=50)

# test individual embedding lookup
emb = embeddings.get_embedding("1")
print(f"embedding for book 1: {emb.shape if emb is not None else 'not found'}")

# test search
results = embeddings.search_books("kafka")
print(f"search results for 'kafka': {len(results)} books")
for book in results[:3]:
    print(f"  {book.title} by {book.author}")

# test properties
print(f"embedding dimension: {embeddings.embedding_dim}")
print(f"total books: {len(embeddings.get_all_books())}")


# test PCA
seed = [embeddings.get_embedding("1"), embeddings.get_embedding("2")]
print(f"seed books: \n {embeddings.books[0]} \n {embeddings.books[1]}")
pca = PCAVibeIdentifier(2)
pca.fit(seed)
pca.find_similar(embeddings, embeddings.books)