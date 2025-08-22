from dataclasses import dataclass

@dataclass
class Book:
    id: str
    title: str
    author: str

@dataclass 
class BookResult:
    book: Book
    similarity: float