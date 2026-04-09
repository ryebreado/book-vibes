# book-vibes

Book recommendation engine that learns a "vibe" from seed books and finds similar ones. Embeddings come from SVD over user ratings (taste signal) and/or SVD over user tags (theme signal); similarity is cosine distance to the seed centroid, optionally with a popularity penalty to suppress gravity-well books.

## Architecture

Three layers, separated so embedding sources and vibe methods can be swapped independently:

- `core.py` — `Book` and `BookResult` dataclasses. Shared types only.
- `embedding_sources.py` — `EmbeddingSource` ABC + implementations that produce per-book vectors. All sources provide the same four-method interface (`get_embedding`, `get_embeddings_batch`, `search_books`, `get_all_books`) plus `embedding_dim`, so callers never have to care which source they're holding.
  - `GoodbooksEmbeddings`: loads `data/raw/ratings.csv`, builds a user×book ratings matrix, runs `TruncatedSVD` on its transpose. Captures a **taste/co-rating** signal ("people who rated X highly also rated Y"). Cached to `data/processed/goodbooks_embeddings_{n_components}.pkl`.
  - `GoodbookTagsEmbeddings`: loads `data/raw/book_tags.csv`, joins `goodreads_book_id → book_id` via `books.csv`, drops tags in the `junk_tags` blocklist (default `DEFAULT_JUNK_TAGS` — library-status / format / sentiment shelves like `to-read`, `owned`, `kindle`, `favorites`), drops tags below `min_tag_count`, builds a sparse book×tag matrix, TF-IDF weights it, runs `TruncatedSVD`. Captures a **theme/content** signal. Cache key includes an MD5 hash of the sorted blocklist so changing it invalidates cleanly: `goodbook_tags_embeddings_{n_components}_min{min_tag_count}_junk{hash8}.pkl`.
  - `HybridEmbeddings`: wraps two other `EmbeddingSource` instances, L2-normalizes each half (so the two sources contribute on equal scale regardless of their internal SVD singular-value magnitudes), and stores `alpha * source_a + (1-alpha) * source_b` per book. Only includes books present in *both* sources. Metadata (title/author) comes from `source_a`. Requires matching `embedding_dim`. Not cached — cheap to rebuild from already-cached sources.
  - `OpenLibraryEmbeddings`: stub, not implemented — references an undefined `Cache` class and will crash if `cache_dir` is passed. Don't treat it as working code.
- `vibe_methods.py` — `VibeIdentifier` ABC + `PCAVibeIdentifier`. `fit()` stores `vibe_centroid = mean(seed_embeddings)` in the **original** embedding space. `find_similar()` computes cosine similarity between the centroid and every catalog book, optionally penalizing by `popularity_weight * log(1 + popularity)` when an aligned `popularity` array is supplied. PCA is still fit on seeds inside `fit()` but *only* so `get_stats()` can report explained-variance ratios as a diagnostic of seed tightness — it is not on the similarity path.
- `test.py` — end-to-end smoke test / example usage. Runs all three pipelines (ratings, tags, hybrid) against the same seed titles so the three signals can be compared side-by-side.

### Search ranking (`EmbeddingSource._match_score`)

`search_books` on every source ranks candidates by a `(tier, title_length)` sort key, lower is better:

- tier 0: exact title match
- tier 1: title starts with query
- tier 2: query appears as substring of title
- tier 3: query appears as substring of author only

Within a tier, shorter titles win (so `"One Day"` beats `"Me Talk Pretty One Day"` for query `"One Day"` — this was a real bug caught during seed-set testing).

## Data

- `data/raw/` — goodbooks-10k CSVs (`books.csv`, `ratings.csv`, `book_tags.csv`, `tags.csv`). Gitignored.
- `data/processed/` — pickle cache of SVD embeddings. Delete the relevant `.pkl` to force regeneration.

Book IDs in goodbooks are 1–10000. `GoodbooksEmbeddings.get_embedding` assumes this range and uses `id - 1` as the row index. `GoodbookTagsEmbeddings` and `HybridEmbeddings` use an explicit `book_id_to_idx` dict, which is more robust — prefer that pattern in any new source. Note that `book_tags.csv` keys on `goodreads_book_id`, not `book_id`, so joins go through `books.csv`.

For popularity-weighted ranking, `books.csv` has a `ratings_count` column (total ratings across all users per book) which is the intended popularity proxy — used as `popularity` input to `find_similar()`.

## Running

```
python test.py
```

⚠️ **Venv is broken**: `.venv/bin/python` is a dangling symlink to a cleaned-up uv-managed python. The known-working invocation during this session was:

```
uv run --with scikit-learn --with pandas --with numpy python test.py
```

When the user rebuilds the venv properly, `python test.py` should work. Don't waste time chasing the venv symlink.

Dependencies: `numpy`, `pandas`, `scikit-learn`, `scipy` (transitively via sklearn, but imported directly in `GoodbookTagsEmbeddings` for `csr_matrix`). See `requirements.txt`.

## Tuned defaults (empirically validated)

These are the settings the recommender was validated against during development. Don't silently change them unless you understand the tradeoff:

- `GoodbookTagsEmbeddings(min_tag_count=20)` — lowered from 100 after seeing that thematic tags like `existentialist`, `absurdist`, `nihilism` were being dropped. At min=20, vocab is ~13k tags. At min=100 it's ~6.4k and loses real signal.
- `GoodbookTagsEmbeddings(junk_tags=DEFAULT_JUNK_TAGS)` — the default blocklist is ~40 tags covering library status (`to-read`, `currently-reading`), ownership (`owned`, `books-i-own`), format (`kindle`, `audiobook`), and pure sentiment (`favorites`). Deliberately does **not** include `classics` or `school` — those carry real (if weak) content signal. Removing the default blocklist drops recommendation quality dramatically (the first tag run before the blocklist returned Bambi and Hound of the Baskervilles for a Kafka vibe).
- `HybridEmbeddings(alpha=0.5)` — equal weight on ratings and tags. Not yet swept; 0.25 / 0.75 are untested but plausible.
- `popularity_weight` on `find_similar`:
  - `0.0` (default) — no penalty, results dominated by popular "gravity well" books (Catch-22, Breakfast at Tiffany's, etc.) that sit near any literary centroid.
  - `~0.02` — mild penalty, reasonable mainstream default.
  - `0.05` — aggressive penalty, produces "deep cuts" mode. Validated on three seed sets (moral-transgression, dark-academia, quiet-realism) and dramatically improved thematic specificity in all three. Example: for a dark-academia seed, unpenalized results drifted to Beloved / Catch-22 / Things Fall Apart; penalized at 0.05 surfaced Fifth Business and Appointment in Samarra at the top — textbook dark-academia novels that a human curator would endorse.

## Project direction

The end goal is a recommender that takes ~5 seed books *from different authors* and returns books matching a broad **thematic vibe** (e.g. "ennui," "absurdist alienation," "cozy melancholy") — not books by the same authors or with the same superficial metadata. The multi-author seed set is the whole point: it's how the user specifies a thematic space wider than any single writer.

**Known limitation of the current sources**: both SVD-over-ratings and SVD-over-tags will pull same-author books to the top of any single-source result list. This is mechanically correct (people who rated Kafka rated other Kafka; people tag Kafka books with Kafka-specific shelves) but it's a failure mode relative to the goal. It reflects a limitation of the goodreads-derived signal, not a bug in the pipeline. `HybridEmbeddings` + `popularity_weight` partially compensate — the hybrid run on a Kafka/Camus/Dostoyevsky seed surfaced Hesse, Bulgakov, and Nabokov (different authors, same vibe) in the top 10, and the popularity penalty further suppressed generic-literary gravity wells.

**Future directions** for better vibe capture:
- **New embedding sources** beyond collaborative-filtering signal — LLM embeddings of book descriptions or prose samples, tropes/theme databases, syllabus co-occurrence. These would capture content directly rather than via reader behavior, and would likely handle aesthetic-register vibes (like "dark academia") better than count-based methods structurally can.
- **De-emphasizing author signal** — either explicit (post-filter at most N books per author) or structural (a source that strips author identity during embedding).
- **Alpha sweeps** on `HybridEmbeddings` — 0.25 / 0.5 / 0.75 comparison would validate whether the 50/50 default is optimal.

When adding a new embedding source, remember that the whole `EmbeddingSource` interface exists so that `vibe_methods.py` doesn't need to care where the vectors came from. New sources should slot in without touching `vibe_methods.py`.

## Current work: web interface

The next phase (starting in a fresh session) is building a web interface around the existing pipeline so users can enter seed books interactively and see recommendations without running `test.py`. The backend is fully functional — the web layer should wrap it, not reinvent it.

Reasonable shape: a small Flask/FastAPI server that constructs the hybrid source on startup (so the SVD caches load once), exposes `/search?q=...` → list of `Book` for seed autocomplete, and `/recommend?seed_ids=...&alpha=...&popularity_weight=...` → list of `BookResult`. Frontend can be a single HTML page with vanilla JS. Nothing about this requires changes to the core pipeline.

## Notes for editing

- **Historical gotcha — don't re-introduce the centroid-in-subspace bug.** An earlier version of `PCAVibeIdentifier.fit()` stored `vibe_centroid = mean(pca.transform(seed_embeddings))`. Because `PCA.fit()` centers data, the mean of the transformed fit-data is numerically zero, making every cosine similarity zero and the "top recommendations" a meaningless `argsort` over an all-zero vector. The current code deliberately keeps the centroid in the original embedding space. If you reintroduce PCA on the similarity path, fit it on the *full catalog* (not just the seeds) so the seed centroid projects to something nonzero.
- `EmbeddingSource._load_from_cache` / `_save_to_cache` provide the caching contract; new sources should use them rather than rolling their own.
- `OpenLibraryEmbeddings.__init__` references an undefined `Cache` class — the stub will crash if `cache_dir` is passed. Not working code.
- `get_stats()` returning `total_explained_variance: 1.0` with almost all weight on the first component is expected for small seed sets (6 books → 5 PCA components capture 100% of 6-point variance by construction). It's a tightness diagnostic, not a quality metric.
- `find_similar()` reports the raw cosine similarity in `BookResult.similarity`, even when `popularity_weight > 0`. The penalty only affects ranking order — the reported number is always the untouched cosine. This is deliberate so the UI can show a consistent "similarity" number across penalty settings.
- `HybridEmbeddings` intersects `book_ids` from both sources. If a new source returns a subset of goodbooks IDs (e.g. an LLM-description source that doesn't cover obscure books), the hybrid will silently shrink to the intersection. The `_build_combined` logging prints the intersection size so this is visible.
