# debug_rag.py
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from src.file_manager import load_file, delete_file
from src import vector_store
from src import rag_pipeline

def main():
    print("=== DEBUG RAG PIPELINE ===")

    # tạo file text test
    p = Path("data/uploads/debug_rag.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "The supreme law of Viet Nam is the Constitution of the Socialist Republic of Viet Nam.",
        encoding="utf-8",
    )

    fid = load_file(str(p))
    print("Loaded file_id:", fid)
    print("Files in DB:", vector_store.list_files())

    question = "What is the supreme law of the Socialist Republic of Vietnam?"
    print("Question:", question)

    result = rag_pipeline.rag_answer(question, k=3)
    print("Answer:", result["answer"])
    print("Num context docs:", len(result["context"]))

    for i, doc in enumerate(result["context"]):
        print(f"--- DOC {i} ---")
        print(doc[:300].replace("\n", " "))
        print("-------------")

    delete_file(fid)
    print("Files after cleanup:", vector_store.list_files())

if __name__ == "__main__":
    main()
