import pandas as pd
import numpy as np
import gradio as gr

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- Load Data ----------------
books = pd.read_csv("books_with_emotions.csv")

books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
books["large_thumbnail"] = np.where(
    books["large_thumbnail"].isna(),
    "cover-not-found.jpg",
    books["large_thumbnail"],
)

# ---------------- Load DB ----------------
embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

db_books = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# ---------------- Logic ----------------
def recommend_books(query, category, tone):
    recs = db_books.similarity_search(query, k=50)

    indices = [
        rec.metadata.get("index")
        for rec in recs
        if rec.metadata.get("index") is not None
    ]

    indices = [i for i in indices if i < len(books)]
    book_recs = books.iloc[indices]

    if category != "All":
        book_recs = book_recs[book_recs["simple_categories"] == category]

    if tone == "Happy":
        book_recs.sort_values(by="joy", ascending=False, inplace=True)
    elif tone == "Surprising":
        book_recs.sort_values(by="surprise", ascending=False, inplace=True)
    elif tone == "Angry":
        book_recs.sort_values(by="anger", ascending=False, inplace=True)
    elif tone == "Suspenseful":
        book_recs.sort_values(by="fear", ascending=False, inplace=True)
    elif tone == "Sad":
        book_recs.sort_values(by="sadness", ascending=False, inplace=True)

    results = []
    for _, row in book_recs.head(16).iterrows():
        caption = f"{row['title']}\n{row['authors']}"
        results.append((row["large_thumbnail"], caption))

    return results

# ---------------- UI ----------------
categories = ["All"] + sorted(books["simple_categories"].unique())
tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]

with gr.Blocks() as demo:
    gr.Markdown("# 📚 Book Recommender")

    query = gr.Textbox(label="Describe a book")

    with gr.Row():
        category = gr.Dropdown(categories, value="All")
        tone = gr.Dropdown(tones, value="All")

    btn = gr.Button("Get Recommendations")
    output = gr.Gallery(columns=4)

    btn.click(recommend_books, [query, category, tone], output)

if __name__ == "__main__":
    demo.launch()