import sqlite3

def init_history_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS file_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    conn.close()