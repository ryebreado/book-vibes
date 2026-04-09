import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from embedding_sources import GoodbooksEmbeddings, GoodbookTagsEmbeddings, HybridEmbeddings
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

print(f"\nTop recommendations (ratings-based):")
for i, result in enumerate(recommendations[:10]):
    print(f"  {i+1}. {result.book.title} by {result.book.author} (sim: {result.similarity:.3f})")


# same vibe, but using tag-based embeddings
print("\n" + "=" * 60)
print("Tag-based embeddings (theme/content signal)")
print("=" * 60)

tag_embeddings = GoodbookTagsEmbeddings("data/raw", n_components=50, min_tag_count=20)

print("Finding seed books in tag source:")
tag_seed_books = []
tag_seed_embeddings = []
for title in seed_book_titles:
    matches = tag_embeddings.search_books(title, max_results=1)
    if matches:
        book = matches[0]
        emb = tag_embeddings.get_embedding(book.id)
        if emb is not None:
            tag_seed_books.append(book)
            tag_seed_embeddings.append(emb)
            print(f"  {book.title} by {book.author}")

tag_vibe = PCAVibeIdentifier(n_components=min(10, len(tag_seed_embeddings) - 1))
tag_vibe.fit(np.array(tag_seed_embeddings))

tag_stats = tag_vibe.get_stats()
print(f"\nTag vibe stats:")
print(f"  explained variance: {tag_stats['total_explained_variance']:.3f}")
print(f"  top 3 components: {tag_stats['explained_variance_ratio'][:3]}")

tag_all_books = tag_embeddings.get_all_books()
tag_all_embeddings = tag_embeddings.get_embeddings_batch([b.id for b in tag_all_books])

tag_seed_ids = {book.id for book in tag_seed_books}
tag_recommendations = tag_vibe.find_similar(
    tag_all_embeddings, tag_all_books, n_results=10, exclude_ids=tag_seed_ids
)

print(f"\nTop recommendations (tag-based):")
for i, result in enumerate(tag_recommendations[:10]):
    print(f"  {i+1}. {result.book.title} by {result.book.author} (sim: {result.similarity:.3f})")


# hybrid: 50/50 mix of taste (ratings) and theme (tags)
print("\n" + "=" * 60)
print("Hybrid embeddings (ratings + tags, alpha=0.5)")
print("=" * 60)

hybrid = HybridEmbeddings(embeddings, tag_embeddings, alpha=0.5)

print("Finding seed books in hybrid source:")
hybrid_seed_books = []
hybrid_seed_embeddings = []
for title in seed_book_titles:
    matches = hybrid.search_books(title, max_results=1)
    if matches:
        book = matches[0]
        emb = hybrid.get_embedding(book.id)
        if emb is not None:
            hybrid_seed_books.append(book)
            hybrid_seed_embeddings.append(emb)
            print(f"  {book.title} by {book.author}")

hybrid_vibe = PCAVibeIdentifier(n_components=min(10, len(hybrid_seed_embeddings) - 1))
hybrid_vibe.fit(np.array(hybrid_seed_embeddings))

hybrid_all_books = hybrid.get_all_books()
hybrid_all_embeddings = hybrid.get_embeddings_batch([b.id for b in hybrid_all_books])

hybrid_seed_ids = {book.id for book in hybrid_seed_books}
hybrid_recommendations = hybrid_vibe.find_similar(
    hybrid_all_embeddings, hybrid_all_books, n_results=10, exclude_ids=hybrid_seed_ids
)

print(f"\nTop recommendations (hybrid):")
for i, result in enumerate(hybrid_recommendations[:10]):
    print(f"  {i+1}. {result.book.title} by {result.book.author} (sim: {result.similarity:.3f})")