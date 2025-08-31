import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

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
seed_book_titles = ["metamorphosis", "stranger", "norwegian wood", "The Great Gatsby", 
"Notes from Underground", "The Trial"]
seed_books = []
seed_embeddings = []

print("Finding seed books:")
for title in seed_book_titles:
    matches = embeddings.search_books(title, max_results=1)
    if matches:
        book = matches[0]
        emb = embeddings.get_embedding(book.id)
        if emb is not None:
            seed_books.append(book)
            seed_embeddings.append(emb)
            print(f"  {book.title} by {book.author}")

# fit the vibe
vibe = PCAVibeIdentifier(n_components=min(10, len(seed_embeddings) - 1))
vibe.fit(np.array(seed_embeddings))

# get stats
stats = vibe.get_stats()
print(f"\nVibe stats:")
print(f"  explained variance: {stats['total_explained_variance']:.3f}")
print(f"  top 3 components: {stats['explained_variance_ratio'][:3]}")

# find similar books
all_books = embeddings.get_all_books()
all_embeddings = embeddings.get_embeddings_batch([b.id for b in all_books])

seed_ids = {book.id for book in seed_books}
recommendations = vibe.find_similar(all_embeddings, all_books, n_results=10, exclude_ids=seed_ids)

print(f"\nTop recommendations:")
for i, result in enumerate(recommendations[:10]):
    print(f"  {i+1}. {result.book.title} by {result.book.author} (sim: {result.similarity:.3f})")