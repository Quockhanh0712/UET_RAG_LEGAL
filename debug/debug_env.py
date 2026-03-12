# debug_env.py
import os
from dotenv import load_dotenv

def main():
    print("=== Load .env ===")
    load_dotenv()

    print("=== GOOGLE_API_KEY ===")
    api_key = os.getenv("GOOGLE_API_KEY")
    print("GOOGLE_API_KEY =", repr(api_key))

    print("=== GEMINI_MODEL ===")
    model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    print("GEMINI_MODEL =", model)

    try:
        from google import genai
        print("Imported google.genai OK")

        if not api_key:
            print("ERROR: GOOGLE_API_KEY is None or empty, không thể tạo client.")
            return

        client = genai.Client(api_key=api_key)
        print("Created genai.Client OK")

        # gọi thử model.list để xem key có hợp lệ không (không tốn tiền)
        models = list(client.models.list())
        print(f"List models OK, số lượng: {len(models)}")
    except Exception as e:
        print("ERROR when using google.genai:", repr(e))

if __name__ == "__main__":
    main()
