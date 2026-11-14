"""Streamlit demo.   streamlit run app.py"""
from pathlib import Path

import streamlit as st
from PIL import Image

from src.config import IMAGES_DIR, INDEX_DIR
from src.search import SearchEngine

st.set_page_config(page_title="Visual Furniture Search", layout="wide")


@st.cache_resource(show_spinner="Loading CLIP model and index ...")
def load_engine(kind: str) -> SearchEngine:
    return SearchEngine.load(INDEX_DIR, IMAGES_DIR, kind=kind)


def show(payload: dict) -> None:
    t = payload["timings_ms"]
    st.caption(f"Encode {t['embed']} ms | vector search {t['search']} ms")
    cols = st.columns(len(payload["results"]) or 1)
    for col, r in zip(cols, payload["results"]):
        col.image(str(Path(IMAGES_DIR) / r["path"]), use_container_width=True)
        col.caption(f"**{r['category']}**  \nscore {r['score']}")


st.title("Visual Furniture Search")
st.write("Find furniture by describing it, or by uploading a photo. "
         "Powered by CLIP embeddings and a FAISS vector index.")

kind = st.sidebar.selectbox("Index type", ["hnsw", "flat"], help="hnsw = approximate, flat = exact")
k = st.sidebar.slider("Results", 3, 12, 6)
engine = load_engine(kind)
st.sidebar.write(f"Gallery: {engine.index.size} images")

text_tab, image_tab = st.tabs(["Search by text", "Search by image"])
with text_tab:
    query = st.text_input("Describe what you are looking for", "grey 3-seater sofa")
    if query:
        show(engine.search_text(query, k))
with image_tab:
    upload = st.file_uploader("Upload a furniture photo", type=["jpg", "jpeg", "png", "webp"])
    if upload:
        image = Image.open(upload).convert("RGB")
        st.image(image, width=240, caption="Your query")
        show(engine.search_image(image, k))
