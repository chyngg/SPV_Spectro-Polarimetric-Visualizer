import sqlite3

def init_history_db():
    conn = sqlite3.connect('history.db')
    c = conn.cursor()
    c.execute("""
        CREATE TABLE IF NOT EXISTS file_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            path TEXT UNIQUE,
            data_type TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
        )
    """)

    c.execute("PRAGMA table_info(file_history)")
    cols = {row[1] for row in c.fetchall()}

    if "data_type" not in cols:
        c.execute("ALTER TABLE file_history ADD COLUMN data_type TEXT DEFAULT 'Unknown'")
    if "timestamp" not in cols:
        c.execute("ALTER TABLE file_history ADD COLUMN timestamp DATETIME NOT NULL DEFAULT (CURRENT_TIMESTAMP)")

    conn.commit()
    conn.close()