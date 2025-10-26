import sqlite3

try:
    conn = sqlite3.connect('test.db')
    cursor = conn.cursor()
    cursor.execute('CREATE TABLE IF NOT EXISTS test (id INTEGER PRIMARY KEY, name TEXT)')
    cursor.execute('INSERT INTO test (name) VALUES (?)', ('SQLite Working!',))
    conn.commit()

    cursor.execute('SELECT * FROM test')
    print("✅ SQLite Connected Successfully! Rows:")
    for row in cursor.fetchall():
        print(row)

    conn.close()
except Exception as e:
    print("❌ SQLite Error:", e)
