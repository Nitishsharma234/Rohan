"""
rag_engine.py — TF-IDF RAG over .txt knowledge files
"""
import os, re, glob, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

DATA_DIR  = os.path.join(os.path.dirname(__file__), "data")
CACHE_PKL = os.path.join(DATA_DIR, "rag_cache.pkl")

def _load_docs():
    docs, sources = [], []
    for fpath in glob.glob(os.path.join(DATA_DIR, "**", "*.txt"), recursive=True):
        try:
            with open(fpath, encoding="utf-8") as f:
                content = f.read()
            blocks = [b.strip() for b in re.split(r"\n{2,}", content) if len(b.strip()) > 40]
            fname  = os.path.relpath(fpath, DATA_DIR)
            docs    += blocks
            sources += [fname] * len(blocks)
        except Exception as e:
            print(f"  [rag] Cannot read {fpath}: {e}")
    return docs, sources

def _build_index(docs):
    vec = TfidfVectorizer(ngram_range=(1,2), sublinear_tf=True)
    return vec, vec.fit_transform(docs)

def _load_or_build():
    if os.path.exists(CACHE_PKL):
        try:
            with open(CACHE_PKL, "rb") as f:
                c = pickle.load(f)
            print(f"  [rag] Loaded cache — {len(c['docs'])} chunks")
            return c["docs"], c["sources"], c["vec"], c["X"]
        except Exception as e:
            print(f"  [rag] Cache failed ({e}), rebuilding…")
    docs, sources = _load_docs()
    if not docs:
        return [], [], None, None
    vec, X = _build_index(docs)
    try:
        with open(CACHE_PKL, "wb") as f:
            pickle.dump({"docs": docs, "sources": sources, "vec": vec, "X": X}, f)
        print(f"  [rag] Built & cached — {len(docs)} chunks")
    except Exception as e:
        print(f"  [rag] Cache save failed: {e}")
    return docs, sources, vec, X

DOCS, SOURCES, VECTORIZER, X_MATRIX = _load_or_build()

def search_rag(query: str, k: int = 4, min_score: float = 0.08):
    if VECTORIZER is None or not DOCS:
        return []
    try:
        q      = VECTORIZER.transform([query])
        scores = cosine_similarity(q, X_MATRIX)[0]
        idxs   = scores.argsort()[-k:][::-1]
        return [{"text": DOCS[i], "source": SOURCES[i], "score": round(float(scores[i]), 3)}
                for i in idxs if scores[i] >= min_score]
    except:
        return []
