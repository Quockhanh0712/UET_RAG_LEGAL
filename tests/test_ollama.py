import ollama

response = ollama.chat(
    model='qwen2.5:3b',
    messages=[{'role': 'user', 'content': 'Xin chào! RAG là gì?'}]
)

print(response['message']['content'])
