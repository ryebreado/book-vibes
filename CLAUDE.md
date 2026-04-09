# book-vibes

Book recommendation engine that learns a "vibe" from seed books and finds similar ones. Uses collaborative-filtering embeddings (SVD over user ratings) + PCA subspace projection for similarity.

## Architecture

Three layers, separated so embedding sources and vibe methods can be swapped independently:

- `core.py` — `Book` and `BookResult` dataclasses. Shared types only.
- `embedding_sources.py` — `EmbeddingSource` ABC + implementations that produce per-book vectors.
  - `GoodbooksEmbeddings`: loads `data/raw/ratings.csv`, builds a user×book ratings matrix, runs `TruncatedSVD` on its transpose to get book embeddings. Results cached to `data/processed/goodbooks_embeddings_{n_components}.pkl`.
  - `OpenLibraryEmbeddings`: stub, not implemented.
- `vibe_methods.py` — `VibeIdentifier` ABC + `PCAVibeIdentifier`: fits PCA on seed book embeddings to define a "vibe subspace," computes the centroid there, and ranks all books by cosine similarity to that centroid in the projected space.
- `test.py` — end-to-end smoke test / example usage (load goodbooks → search seeds → fit vibe → print top recommendations).

## Data

- `data/raw/` — goodbooks-10k CSVs (`books.csv`, `ratings.csv`, `book_tags.csv`, `tags.csv`). Gitignored.
- `data/processed/` — pickle cache of SVD embeddings. Delete the `.pkl` to force regeneration.

Book IDs in goodbooks are 1–10000; `GoodbooksEmbeddings.get_embedding` assumes this range and uses `id - 1` as the row index into `book_embeddings`.

## Running

```
python test.py
```

No test framework — `test.py` is a script, not pytest. First run builds the SVD cache (slow); subsequent runs load from pickle.

Dependencies: `numpy`, `pandas`, `scikit-learn` (see `requirements.txt`). Venv lives in `.venv/`.

## Notes for editing

- `PCAVibeIdentifier.find_similar` currently recomputes `projected_books` and `similarities` twice and prints debug stats — leftover from debugging, safe to clean up if asked.
- `EmbeddingSource._load_from_cache` / `_save_to_cache` provide the caching contract; new sources should use them rather than rolling their own.
- `OpenLibraryEmbeddings.__init__` references an undefined `Cache` class — the stub will crash if `cache_dir` is passed. Don't treat it as working code.
