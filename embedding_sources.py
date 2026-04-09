from abc import ABC, abstractmethod
from typing import List, Optional, Iterable
import hashlib
import numpy as np
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfTransformer
import pandas as pd
from core import Book
import pickle
from pathlib import Path


# Default blocklist: library-status, format, and sentiment tags that carry
# no content signal. Content descriptors like "classics" and "school" are
# deliberately NOT here — they convey a real (if weak) signal.
DEFAULT_JUNK_TAGS = frozenset({
    # library status
    "to-read", "read", "currently-reading", "reading",
    "to-buy", "wishlist", "wish-list", "to-read-owned",
    "dnf", "did-not-finish", "abandoned", "unfinished",
    "re-read", "reread", "re-reads", "rereads",
    "default",
    # ownership
    "owned", "owned-books", "books-i-own", "i-own", "my-books",
    "my-library", "library", "home-library",
    # format
    "kindle", "audiobook", "audiobooks", "audio", "audio-books",
    "ebook", "e-book", "ebooks", "hardcover", "paperback",
    # sentiment only
    "favorites", "favourites", "favorite", "favourite",
    "all-time-favorites", "all-time-favourites", "favorite-books",
    # reading context (no content)
    "book-club",
})

class EmbeddingSource(ABC):
    def __init__(self, cache_dir: str = "data/processed"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    @abstractmethod
    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        """Get embedding for a specific book. Returns None if not found."""
        pass
    
    @abstractmethod
    def search_books(self, query: str, max_results: int = 10) -> List[Book]:
        """Fuzzy search for books by title/author."""
        pass
    
    @abstractmethod
    def get_all_books(self) -> List[Book]:
        """Get all available books (for finding recommendations)."""
        pass
    
    @abstractmethod
    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        """Get embeddings for multiple books efficiently."""
        pass
    
    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings."""
        pass

    def _get_cache_path(self, cache_name: str) -> Path:
        """Generate cache file path"""
        return self.cache_dir / f"{cache_name}.pkl"
    
    def _load_from_cache(self, cache_name: str) -> Optional[dict]:
        """Load data from cache if it exists"""
        cache_file = self._get_cache_path(cache_name)
        if cache_file.exists():
            print(f"Loading cached {cache_name}...")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _save_to_cache(self, cache_name: str, data: dict) -> None:
        """Save data to cache"""
        cache_file = self._get_cache_path(cache_name)
        print(f"Caching {cache_name}...")
        with open(cache_file, 'wb') as f:
            pickle.dump(data, f)

class GoodbooksEmbeddings(EmbeddingSource):
    def __init__(self, data_path: str, n_components: int = 50, cache_dir: str = "data/processed"):
        super().__init__(cache_dir)
        self.data_path = data_path
        self.n_components = n_components
        self._load_and_process()
    
    def _load_and_process(self):
        cache_name = f"goodbooks_embeddings_{self.n_components}"

        cached_data = self._load_from_cache(cache_name)

        # check if data cached already, o/w generate with SVD
        if cached_data:
            self.book_embeddings = cached_data['embeddings']
            self.book_ids = cached_data['book_ids']
            print(f"Loaded embeddings for {len(self.book_ids)} books")
        else:
            print("Loading data...")  # helpful to see progress
            ratings = pd.read_csv(f"{self.data_path}/ratings.csv")
            
            unique_books = ratings['book_id'].unique()
            print(f"books with ratings: {len(unique_books)}")

            print("Creating ratings matrix...")
            ratings_matrix = ratings.pivot(index='user_id', columns='book_id', values='rating').fillna(0)
            
            print("Running SVD...")
            svd = TruncatedSVD(n_components=self.n_components)
            self.book_embeddings = svd.fit_transform(ratings_matrix.T)
            
            # check if we have all books
            self.book_ids = ratings_matrix.columns.tolist()
            print(f"Generated embeddings for {len(self.book_ids)} books")

            self._save_to_cache(cache_name, {
                'embeddings': self.book_embeddings,
                'book_ids': self.book_ids
            })
        
        books = pd.read_csv(f"{self.data_path}/books.csv")
        self.books = books.set_index('book_id')
    
    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        book_id_int = int(book_id)
        if 1 <= book_id_int <= 10000: 
            idx = book_id_int - 1  # convert to 0-indexed
            return self.book_embeddings[idx]
        return None
    
    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        """Get embeddings for MULTIPLE books efficiently"""
        # vectorized lookup
        embeddings = []
        for book_id in book_ids:
            emb = self.get_embedding(book_id)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.array([])  # empty array if no valid books
        
        return np.vstack(embeddings)  # stack into 2D array: n_books × n_components
    
    def search_books(self, query: str, max_results: int = 10) -> List[Book]:
        """Fuzzy search books by title/author"""
        # this searches the METADATA, not the embeddings
        # for inputing books
        # use the books.csv data for fuzzy matching
        query_lower = query.lower()
        matches = []
        
        for book_id, book_info in self.books.iterrows():
            title = str(book_info['title']).lower()
            author = str(book_info['authors']).lower()  
            
            # simple fuzzy matching
            if query_lower in title or query_lower in author:
                matches.append(Book(
                    id=str(book_id),
                    title=book_info['title'],
                    author=book_info['authors']
                ))
        
        return matches[:max_results]
    
    def get_all_books(self) -> List[Book]:
        """Return all available books"""
        # needed for finding recommendations from the full catalog
        return [
            Book(id=str(book_id), title=row['title'], author=row['authors'])
            for book_id, row in self.books.iterrows()
            if book_id in self.book_ids  # only books that have embeddings
        ]
    
    @property
    def embedding_dim(self) -> int:
        """Dimensionality of embeddings"""
        return self.n_components

class GoodbookTagsEmbeddings(EmbeddingSource):
    """Embeddings built from user-applied tags (shelves) rather than ratings.

    Captures a theme/content signal — books cluster if users describe them
    with the same tags — as opposed to the co-rating signal in GoodbooksEmbeddings.
    """
    def __init__(self, data_path: str, n_components: int = 50,
                 min_tag_count: int = 100,
                 junk_tags: Optional[Iterable[str]] = None,
                 cache_dir: str = "data/processed"):
        super().__init__(cache_dir)
        self.data_path = data_path
        self.n_components = n_components
        self.min_tag_count = min_tag_count
        self.junk_tags = frozenset(
            DEFAULT_JUNK_TAGS if junk_tags is None else junk_tags
        )
        self._load_and_process()

    def _load_and_process(self):
        # hash the blocklist into the cache key so changes invalidate cleanly
        junk_hash = hashlib.md5(
            ",".join(sorted(self.junk_tags)).encode()
        ).hexdigest()[:8]
        cache_name = (
            f"goodbook_tags_embeddings_{self.n_components}"
            f"_min{self.min_tag_count}_junk{junk_hash}"
        )

        cached_data = self._load_from_cache(cache_name)

        if cached_data:
            self.book_embeddings = cached_data['embeddings']
            self.book_ids = cached_data['book_ids']
            print(f"Loaded tag embeddings for {len(self.book_ids)} books")
        else:
            print("Loading books metadata for ID mapping...")
            books_df = pd.read_csv(f"{self.data_path}/books.csv")
            # book_tags.csv uses goodreads_book_id; map it to internal book_id
            gr_to_book = dict(zip(books_df['goodreads_book_id'], books_df['book_id']))

            print("Loading tags.csv and book_tags.csv...")
            tags_df = pd.read_csv(f"{self.data_path}/tags.csv")
            book_tags = pd.read_csv(f"{self.data_path}/book_tags.csv")
            book_tags['book_id'] = book_tags['goodreads_book_id'].map(gr_to_book)
            book_tags = book_tags.dropna(subset=['book_id'])
            book_tags['book_id'] = book_tags['book_id'].astype(int)

            # drop blocklisted shelves (library status / format / sentiment)
            tag_name = dict(zip(tags_df['tag_id'], tags_df['tag_name']))
            blocked_tag_ids = {
                tid for tid, name in tag_name.items()
                if str(name).strip().lower() in self.junk_tags
            }
            before = len(book_tags)
            book_tags = book_tags[~book_tags['tag_id'].isin(blocked_tag_ids)]
            print(f"dropped {before - len(book_tags):,} rows from {len(blocked_tag_ids)} blocklisted tags")

            # drop rare tags — most of the 34k tags are noise (typos, junk shelves)
            tag_totals = book_tags.groupby('tag_id')['count'].sum()
            keep_tags = tag_totals[tag_totals >= self.min_tag_count].index
            book_tags = book_tags[book_tags['tag_id'].isin(keep_tags)]
            print(f"kept {len(keep_tags)} tags after min_count={self.min_tag_count} filter")

            # build sparse book × tag matrix
            print("Building sparse book × tag matrix...")
            book_ids_sorted = sorted(book_tags['book_id'].unique())
            tag_ids_sorted = sorted(keep_tags)
            book_idx = {b: i for i, b in enumerate(book_ids_sorted)}
            tag_idx = {t: i for i, t in enumerate(tag_ids_sorted)}

            rows = book_tags['book_id'].map(book_idx).values
            cols = book_tags['tag_id'].map(tag_idx).values
            data = book_tags['count'].values.astype(float)

            matrix = csr_matrix(
                (data, (rows, cols)),
                shape=(len(book_ids_sorted), len(tag_ids_sorted))
            )
            print(f"matrix shape: {matrix.shape}, nnz: {matrix.nnz}")

            # TF-IDF weighting: downweights tags like "to-read" that appear everywhere
            print("Applying TF-IDF weighting...")
            tfidf = TfidfTransformer()
            matrix = tfidf.fit_transform(matrix)

            print("Running SVD...")
            svd = TruncatedSVD(n_components=self.n_components)
            self.book_embeddings = svd.fit_transform(matrix)
            self.book_ids = [str(b) for b in book_ids_sorted]
            print(f"Generated tag embeddings for {len(self.book_ids)} books")

            self._save_to_cache(cache_name, {
                'embeddings': self.book_embeddings,
                'book_ids': self.book_ids,
            })

        # lookup index — more robust than the 1-10000 assumption in GoodbooksEmbeddings
        self.book_id_to_idx = {bid: i for i, bid in enumerate(self.book_ids)}

        books = pd.read_csv(f"{self.data_path}/books.csv")
        self.books = books.set_index('book_id')

    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        idx = self.book_id_to_idx.get(str(book_id))
        if idx is None:
            return None
        return self.book_embeddings[idx]

    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        embeddings = []
        for book_id in book_ids:
            emb = self.get_embedding(book_id)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return np.array([])

        return np.vstack(embeddings)

    def search_books(self, query: str, max_results: int = 10) -> List[Book]:
        query_lower = query.lower()
        matches = []

        for book_id, book_info in self.books.iterrows():
            if str(book_id) not in self.book_id_to_idx:
                continue
            title = str(book_info['title']).lower()
            author = str(book_info['authors']).lower()

            if query_lower in title or query_lower in author:
                matches.append(Book(
                    id=str(book_id),
                    title=book_info['title'],
                    author=book_info['authors']
                ))

        return matches[:max_results]

    def get_all_books(self) -> List[Book]:
        return [
            Book(id=str(book_id), title=row['title'], author=row['authors'])
            for book_id, row in self.books.iterrows()
            if str(book_id) in self.book_id_to_idx
        ]

    @property
    def embedding_dim(self) -> int:
        return self.n_components


class HybridEmbeddings(EmbeddingSource):
    """Combine two embedding sources into a single per-book vector.

    Each half is L2-normalized (so the two sources contribute on equal scale
    regardless of their internal SVD singular-value magnitudes), then mixed:
        hybrid = alpha * source_a + (1 - alpha) * source_b

    Only books present in *both* sources are included. Metadata comes from
    source_a.

    Requires both sources to have the same embedding_dim.
    """
    def __init__(self, source_a: EmbeddingSource, source_b: EmbeddingSource,
                 alpha: float = 0.5, cache_dir: str = "data/processed"):
        super().__init__(cache_dir)
        if source_a.embedding_dim != source_b.embedding_dim:
            raise ValueError(
                f"embedding_dim mismatch: {source_a.embedding_dim} vs "
                f"{source_b.embedding_dim}. Hybrid requires matching dims."
            )
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"alpha must be in [0, 1], got {alpha}")

        self.source_a = source_a
        self.source_b = source_b
        self.alpha = alpha
        self._build_combined()

    @staticmethod
    def _l2_normalize(x: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms[norms == 0] = 1.0  # avoid div-by-zero on any pathological zero rows
        return x / norms

    def _build_combined(self):
        print(f"Building hybrid embeddings (alpha={self.alpha})...")

        books_a = {b.id: b for b in self.source_a.get_all_books()}
        books_b_ids = {b.id for b in self.source_b.get_all_books()}

        # intersection, in a stable order for reproducibility
        common_ids = sorted(books_a.keys() & books_b_ids, key=lambda s: int(s))
        print(f"  source_a: {len(books_a):,} books, source_b: {len(books_b_ids):,} books")
        print(f"  intersection: {len(common_ids):,} books")

        emb_a = self.source_a.get_embeddings_batch(common_ids)
        emb_b = self.source_b.get_embeddings_batch(common_ids)

        if emb_a.shape != emb_b.shape:
            raise RuntimeError(
                f"batch shape mismatch: {emb_a.shape} vs {emb_b.shape}. "
                "A source may be returning None for some common IDs."
            )

        emb_a = self._l2_normalize(emb_a)
        emb_b = self._l2_normalize(emb_b)

        self.book_embeddings = self.alpha * emb_a + (1.0 - self.alpha) * emb_b
        self.book_ids = common_ids
        self.book_id_to_idx = {bid: i for i, bid in enumerate(common_ids)}
        self.books = {bid: books_a[bid] for bid in common_ids}

    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        idx = self.book_id_to_idx.get(str(book_id))
        if idx is None:
            return None
        return self.book_embeddings[idx]

    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        embeddings = []
        for book_id in book_ids:
            emb = self.get_embedding(book_id)
            if emb is not None:
                embeddings.append(emb)

        if not embeddings:
            return np.array([])

        return np.vstack(embeddings)

    def search_books(self, query: str, max_results: int = 10) -> List[Book]:
        query_lower = query.lower()
        matches = []
        for book_id in self.book_ids:
            book = self.books[book_id]
            if query_lower in book.title.lower() or query_lower in book.author.lower():
                matches.append(book)
        return matches[:max_results]

    def get_all_books(self) -> List[Book]:
        return [self.books[bid] for bid in self.book_ids]

    @property
    def embedding_dim(self) -> int:
        return self.source_a.embedding_dim


class OpenLibraryEmbeddings(EmbeddingSource):
    def __init__(self, api_key: str, cache_dir: str = None, n_components=50):
        super().__init__(cache_dir)
        self.api_key = api_key
        self.cache = Cache(cache_dir) if cache_dir else None
    
    def _load_and_process(self):
        pass

    def get_embedding(self, book_id: str) -> Optional[np.ndarray]:
        pass

    def search_books(self, query: str, max_results: int = 10) -> List[Book]:
        pass 

    def get_all_books(self) -> List[Book]:
        """Get all available books (for finding recommendations)."""
        pass
    
    def get_embeddings_batch(self, book_ids: List[str]) -> np.ndarray:
        """Get embeddings for multiple books efficiently."""
        embeddings = []
        for book_id in book_ids:
            emb = self.get_embedding(book_id)
            if emb is not None:
                embeddings.append(emb)
        
        if not embeddings:
            return np.array([])  # empty array if no valid books
        
        return np.vstack(embeddings)  # stack into 2D array: n_books × n_components