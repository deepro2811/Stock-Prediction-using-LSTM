import sqlite3

def connect_db():
    return sqlite3.connect('predictions.db')

def create_table():
    with connect_db() as conn:
        conn.execute('''CREATE TABLE IF NOT EXISTS predictions
                        (id INTEGER PRIMARY KEY AUTOINCREMENT,
                         symbol TEXT NOT NULL,
                         predicted_price REAL NOT NULL,
                         date TEXT NOT NULL,
                         timestamp DATETIME DEFAULT CURRENT_TIMESTAMP)''')

def store_prediction(symbol, predicted_price, date):
    with connect_db() as conn:
        conn.execute('INSERT INTO predictions (symbol, predicted_price, date) VALUES (?, ?, ?)',
                     (symbol, predicted_price, date))
        conn.commit()

def query_db(query, args=(), one=False):
    with connect_db() as conn:
        conn.row_factory = sqlite3.Row
        cur = conn.execute(query, args)
        rv = cur.fetchall()
        return (rv[0] if rv else None) if one else rv
