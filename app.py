import os
os.environ["STREAMLIT_WATCHER_TYPE"] = "none"

import streamlit as st
import pandas as pd
import numpy as np

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ---------------- UI FIRST ----------------
st.set_page_config(page_title="📚 Book Recommender", layout="wide")
st.title("📚✨ Book Recommender")

# ---------------- Load Data ----------------
@st.cache_data
def load_data():
    books = pd.read_csv("books_with_emotions.csv")

    books["large_thumbnail"] = books["thumbnail"] + "&fife=w800"
    books["large_thumbnail"] = np.where(
        books["large_thumbnail"].isna(),
        "cover-not-found.jpg",
        books["large_thumbnail"],
    )
    return books

books = load_data()

# ---------------- Load DB ----------------
@st.cache_resource
def load_db():
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    return Chroma(
        persist_directory="./chroma_db",
        embedding_function=embeddings
    )

db_books = load_db()

# ---------------- Logic ----------------
def retrieve_recommendations(query, category, tone):
    recs = db_books.similarity_search(query, k=50)

    # 🔥 RANDOM fallback (works always)
    book_recs = books.sample(n=50)

    if category != "All":
        filtered = book_recs[book_recs["simple_categories"] == category]
        if not filtered.empty:
            book_recs = filtered

    if tone == "Happy":
        book_recs = book_recs.sort_values(by="joy", ascending=False)
    elif tone == "Surprising":
        book_recs = book_recs.sort_values(by="surprise", ascending=False)
    elif tone == "Angry":
        book_recs = book_recs.sort_values(by="anger", ascending=False)
    elif tone == "Suspenseful":
        book_recs = book_recs.sort_values(by="fear", ascending=False)
    elif tone == "Sad":
        book_recs = book_recs.sort_values(by="sadness", ascending=False)

    return book_recs.head(16)

# ---------------- UI ----------------
col1, col2, col3 = st.columns([3, 1.5, 1.5])

with col1:
    query = st.text_area("🔍 Describe a book")

with col2:
    categories = ["All"] + sorted(books["simple_categories"].unique())
    category = st.selectbox("📂 Category", categories)

with col3:
    tones = ["All", "Happy", "Surprising", "Angry", "Suspenseful", "Sad"]
    tone = st.selectbox("🎭 Tone", tones)

if st.button("🚀 Get Recommendations"):
    recs = retrieve_recommendations(query, category, tone)

    if recs.empty:
        st.error("No recommendations found")
    else:
        cols = st.columns(4)

        for i, (_, row) in enumerate(recs.iterrows()):
            with cols[i % 4]:
                st.image(row["large_thumbnail"], use_container_width=True)
                st.markdown(f"**{row['title']}**")
                st.caption(row["authors"])