import os
import sys

# Ensure project root is in sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from db.connection import init_db, get_db_connection

def verify() -> None:
    print("Initializing database...")
    init_db()
    print("Database initialized successfully.")

    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = [row["name"] for row in cursor.fetchall()]
    print("Tables created in database:")
    for t in tables:
        print(f"  - {t}")
    conn.close()

if __name__ == "__main__":
    verify()
