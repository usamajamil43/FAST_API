from fastapi import FastAPI
app = FastAPI()
@app.get("/")
async def root():
    return {"message": "Hello World"}
@app.get("/items/{item_id}")
# Define a function that will accept the variable
def read_item(item_id: int):
    return {"item_id": item_id}
from fastapi import FastAPI

app = FastAPI()

@app.get("/books/")
async def get_books(title=None):
    # Sample data
    books_data = [
        {"title": "Dune", "author": "Frank Herbert"},
        {"title": "1984", "author": "George Orwell"},
        {"title": "Brave New World", "author": "Aldous Huxley"},
        {"title": "Foundation", "author": "Isaac Asimov"}
    ]

    # Filter by title if provided
    if title:
        books_data = [book for book in books_data if title.lower() in book["title"].lower()]
    
    return books_data
