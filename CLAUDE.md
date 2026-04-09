# book-vibes

Book recommendation engine that learns a "vibe" from seed books and finds similar ones. Embeddings come from SVD over either user ratings (taste signal) or user tags (theme signal); similarity is cosine distance to the seed centroid in the original embedding space.

## Architecture

Three layers, separated so embedding sources and vibe methods can be swapped independently:

- `core.py` — `Book` and `BookResult` dataclasses. Shared types only.
- `embedding_sources.py` — `EmbeddingSource` ABC + implementations that produce per-book vectors.
  - `GoodbooksEmbeddings`: loads `data/raw/ratings.csv`, builds a user×book ratings matrix, runs `TruncatedSVD` on its transpose to get book embeddings. Captures a **taste/co-rating** signal. Cached to `data/processed/goodbooks_embeddings_{n_components}.pkl`.
  - `GoodbookTagsEmbeddings`: loads `data/raw/book_tags.csv`, joins `goodreads_book_id → book_id` via `books.csv`, filters tags by `min_tag_count`, builds a sparse book×tag matrix, TF-IDF weights it, runs `TruncatedSVD`. Captures a **theme/content** signal — two books cluster if users describe them with the same shelves. Cached to `data/processed/goodbook_tags_embeddings_{n_components}_min{min_tag_count}.pkl`.
  - `OpenLibraryEmbeddings`: stub, not implemented.
- `vibe_methods.py` — `VibeIdentifier` ABC + `PCAVibeIdentifier`: stores `vibe_centroid = mean(seed_embeddings)` in the **original** embedding space and ranks all books by cosine similarity to that centroid. PCA is still fit on seeds inside `fit()` but only so `get_stats()` can report explained-variance ratios as a diagnostic of how tightly the seeds cluster — it is not on the similarity path.
- `test.py` — end-to-end smoke test / example usage (load goodbooks → search seeds → fit vibe → print top recommendations).

## Data

- `data/raw/` — goodbooks-10k CSVs (`books.csv`, `ratings.csv`, `book_tags.csv`, `tags.csv`). Gitignored.
- `data/processed/` — pickle cache of SVD embeddings. Delete the `.pkl` to force regeneration.

Book IDs in goodbooks are 1–10000; `GoodbooksEmbeddings.get_embedding` assumes this range and uses `id - 1` as the row index into `book_embeddings`. `GoodbookTagsEmbeddings` uses an explicit `book_id_to_idx` dict instead, which is more robust — prefer that pattern in any new source. Note that `book_tags.csv` keys on `goodreads_book_id`, not `book_id`, so joins go through `books.csv`.

## Running

```
python test.py
```

No test framework — `test.py` is a script, not pytest. First run builds the SVD cache (slow); subsequent runs load from pickle.

Dependencies: `numpy`, `pandas`, `scikit-learn` (see `requirements.txt`). Venv lives in `.venv/`.

## Project direction

The end goal is a recommender that takes ~5 seed books *from different authors* and returns books matching a broad **thematic vibe** (e.g. "ennui," "absurdist alienation," "cozy melancholy") — not books by the same authors or with the same superficial metadata. The seed set being multi-author is the whole point: it's the user's way of specifying a thematic space wider than any single writer.

**Known limitation of the current sources**: both SVD-over-ratings and SVD-over-tags will pull same-author books to the top of any result list. This is mechanically correct (people who rated Kafka rated other Kafka; people tag Kafka books with Kafka-specific shelves) but it's a failure mode relative to the goal. It reflects a limitation of the goodreads-derived signal, not a bug in the pipeline.

**Future directions** for better vibe capture:
- **New embedding sources** beyond collaborative-filtering signal — e.g. LLM embeddings of book descriptions or prose samples, tropes/theme databases, syllabus co-occurrence. These would capture content directly rather than via reader behavior.
- **De-emphasizing author signal** — either explicit (filter same-author results) or structural (a source that intentionally strips author identity during embedding).
- **Hybrid combinations** to balance multiple signals (see `HybridEmbeddings`).

When adding a new embedding source, remember that the whole `EmbeddingSource` interface exists so that the vibe-finding logic in `vibe_methods.py` doesn't need to care where the vectors came from. New sources should slot in without touching `vibe_methods.py`.

## Notes for editing

- **Historical gotcha — don't re-introduce the centroid-in-subspace bug.** An earlier version of `PCAVibeIdentifier.fit()` stored `vibe_centroid = mean(pca.transform(seed_embeddings))`. Because `PCA.fit()` centers data, the mean of the transformed fit-data is numerically zero, making every cosine similarity zero and the "top recommendations" a meaningless `argsort` over an all-zero vector. The current code deliberately keeps the centroid in the original embedding space. If you reintroduce PCA on the similarity path, fit it on the *full catalog* (not just the seeds) so the seed centroid projects to something nonzero.
- `EmbeddingSource._load_from_cache` / `_save_to_cache` provide the caching contract; new sources should use them rather than rolling their own.
- `OpenLibraryEmbeddings.__init__` references an undefined `Cache` class — the stub will crash if `cache_dir` is passed. Don't treat it as working code.
- `test.py` runs both `GoodbooksEmbeddings` and `GoodbookTagsEmbeddings` end-to-end against the same seed titles so the two signals can be compared side-by-side.
- `get_stats()` returning `total_explained_variance: 1.0` with almost all weight on the first component is expected for small seed sets (6 books → 5 PCA components capture 100% of 6-point variance by construction). It's a tightness diagnostic, not a quality metric.
