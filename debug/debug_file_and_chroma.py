# debug_file_and_chroma.py
from dotenv import load_dotenv
load_dotenv()

from pathlib import Path
from src.file_manager import load_file, delete_file
from src import vector_store

def main():
    print("=== DEBUG FILE + CHROMA ===")

    # tạo file text test
    p = Path("data/uploads/debug_direct.txt")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(
        "RAG la ky thuat ket hop LLM voi du lieu rieng cua nguoi dung.",
        encoding="utf-8",
    )
    print("Test file path:", p, "exists:", p.exists())

    # gọi load_file
    fid = load_file(str(p))
    print("load_file returned file_id:", fid)

    files = vector_store.list_files()
    print("Files in Chroma after load_file:", files)

    # thử query trực tiếp
    res = vector_store.query_documents("RAG la gi?", k=3)
    docs = res["documents"][0]
    print("Num docs returned by query_documents:", len(docs))

    # dọn dẹp
    delete_file(fid)
    print("Files in Chroma after delete_file:", vector_store.list_files())

if __name__ == "__main__":
    main()
