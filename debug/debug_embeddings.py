# debug_embeddings.py
from dotenv import load_dotenv
load_dotenv()

from src.embeddings import get_model, embed_texts

def main():
    print("=== DEBUG EMBEDDINGS ===")
    model = get_model()
    print("Model loaded:", type(model), "name:", model.__class__.__name__)

    texts = ["xin chao", "RAG la gi?"]
    vecs = embed_texts(texts)
    print("Num vectors:", len(vecs))
    print("Vector dim:", len(vecs[0]) if vecs else 0)

if __name__ == "__main__":
    main()
