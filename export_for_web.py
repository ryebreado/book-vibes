"""Export cached embeddings and book metadata to compact JSON for the web UI.

Produces:
  web_export/books.json      — [{id, title, author, image_url, popularity,
                                  average_rating, year}, ...]
  web_export/embeddings.json — {id: [float, ...], ...}

`popularity` is `ratings_count` from books.csv (the proxy used by
`find_similar`'s popularity penalty). `average_rating` and `year` are
included so the UI can show and optionally sort on them.

Both files are written with no pretty-printing. Embedding floats are rounded
to 4 decimal places to keep the payload small.

Works with any EmbeddingSource subclass — pass --source to pick one.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from embedding_sources import (
    EmbeddingSource,
    GoodbooksEmbeddings,
    GoodbookTagsEmbeddings,
    HybridEmbeddings,
)


def build_source(name: str, data_path: str, n_components: int,
                 min_tag_count: int, alpha: float) -> EmbeddingSource:
    if name == "goodbooks":
        return GoodbooksEmbeddings(data_path, n_components=n_components)
    if name == "tags":
        return GoodbookTagsEmbeddings(
            data_path, n_components=n_components, min_tag_count=min_tag_count
        )
    if name == "hybrid":
        a = GoodbooksEmbeddings(data_path, n_components=n_components)
        b = GoodbookTagsEmbeddings(
            data_path, n_components=n_components, min_tag_count=min_tag_count
        )
        return HybridEmbeddings(a, b, alpha=alpha)
    raise ValueError(f"unknown source: {name}")


def export(source: EmbeddingSource, data_path: str, out_dir: Path,
           decimals: int = 4) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # image_url / popularity / ratings / year live in books.csv and aren't
    # surfaced by the EmbeddingSource interface — load them directly and
    # join by book_id.
    books_csv = pd.read_csv(f"{data_path}/books.csv")
    meta_by_id = {}
    for row in books_csv.itertuples(index=False):
        meta_by_id[str(row.book_id)] = {
            "image_url": row.image_url if pd.notna(row.image_url) else None,
            "popularity": int(row.ratings_count) if pd.notna(row.ratings_count) else 0,
            "average_rating": (
                round(float(row.average_rating), 2)
                if pd.notna(row.average_rating) else None
            ),
            "year": (
                int(row.original_publication_year)
                if pd.notna(row.original_publication_year) else None
            ),
        }

    all_books = source.get_all_books()
    book_ids = [b.id for b in all_books]
    embeddings_matrix = source.get_embeddings_batch(book_ids)

    if len(embeddings_matrix) != len(all_books):
        raise RuntimeError(
            f"embedding count {len(embeddings_matrix)} != book count {len(all_books)}; "
            "source returned None for some IDs"
        )

    empty_meta = {"image_url": None, "popularity": 0,
                  "average_rating": None, "year": None}
    books_payload = [
        {
            "id": b.id,
            "title": b.title,
            "author": b.author,
            **meta_by_id.get(str(b.id), empty_meta),
        }
        for b in all_books
    ]

    embeddings_payload = {
        b.id: [round(float(x), decimals) for x in embeddings_matrix[i]]
        for i, b in enumerate(all_books)
    }

    books_path = out_dir / "books.json"
    embeddings_path = out_dir / "embeddings.json"

    with open(books_path, "w") as f:
        json.dump(books_payload, f, separators=(",", ":"), ensure_ascii=False)
    with open(embeddings_path, "w") as f:
        json.dump(embeddings_payload, f, separators=(",", ":"))

    print(f"wrote {len(books_payload):,} books → {books_path} "
          f"({books_path.stat().st_size / 1024:.1f} KB)")
    print(f"wrote {len(embeddings_payload):,} embeddings → {embeddings_path} "
          f"({embeddings_path.stat().st_size / 1024:.1f} KB)")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--source",
        choices=["goodbooks", "tags", "hybrid"],
        default="goodbooks",
        help="which EmbeddingSource to export (default: goodbooks)",
    )
    p.add_argument("--data-path", default="data/raw")
    p.add_argument("--out-dir", default="web_export")
    p.add_argument("--n-components", type=int, default=50)
    p.add_argument("--min-tag-count", type=int, default=20,
                   help="only used for tags / hybrid sources")
    p.add_argument("--alpha", type=float, default=0.5,
                   help="only used for hybrid source")
    p.add_argument("--decimals", type=int, default=4)
    args = p.parse_args()

    source = build_source(
        args.source, args.data_path, args.n_components,
        args.min_tag_count, args.alpha,
    )
    export(source, args.data_path, Path(args.out_dir), decimals=args.decimals)


if __name__ == "__main__":
    main()
