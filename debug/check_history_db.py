"""
Quick test to verify chat history database
"""
import sqlite3
from pathlib import Path

DB_PATH = Path("data/chat_history.db")

if not DB_PATH.exists():
    print("❌ Database not found!")
    print(f"Expected at: {DB_PATH.absolute()}")
else:
    print(f"✅ Database found: {DB_PATH.absolute()}")
    print(f"Size: {DB_PATH.stat().st_size} bytes\n")
    
    conn = sqlite3.connect(str(DB_PATH))
    
    # Check conversations
    conv_count = conn.execute("SELECT COUNT(*) FROM conversations").fetchone()[0]
    print(f"📊 Conversations: {conv_count}")
    
    # Check messages
    msg_count = conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
    print(f"💬 Messages: {msg_count}\n")
    
    if conv_count > 0:
        print("📝 Recent Conversations:")
        for row in conn.execute("SELECT id, title, created_at FROM conversations ORDER BY created_at DESC LIMIT 5"):
            print(f"  - ID: {row[0]} | {row[1]} | {row[2]}")
    
    if msg_count > 0:
        print("\n💬 Recent Messages:")
        for row in conn.execute("SELECT role, content, created_at FROM messages ORDER BY created_at DESC LIMIT 10"):
            content = row[1][:80] + "..." if len(row[1]) > 80 else row[1]
            print(f"  - [{row[0]}] {content}")
            print(f"    Time: {row[2]}")
    
    conn.close()
    
    if conv_count == 0 and msg_count == 0:
        print("\n⚠️ Database is empty!")
        print("This might mean:")
        print("1. No conversations yet - try chatting")
        print("2. Code may have errors - check app logs")
        print("3. Session ID mismatch - restart app")
