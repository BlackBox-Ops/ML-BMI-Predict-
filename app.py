# app.py
from flask import Flask, render_template, request, redirect, url_for
import psycopg2

app = Flask(__name__)

# PostgreSQL connection setup
conn = psycopg2.connect(
    dbname="predictions",
    user="postgres",
    password="admin",
    host="localhost",
    port="5432"
)

@app.route('/')
def index():
    cur = conn.cursor()
    cur.execute("SELECT * FROM predictions;")
    rows = cur.fetchall()
    cur.close()
    return render_template('index.html', rows=[{
        'id': r[0], 'gender': r[1], 'height': r[2], 'weight': r[3], 'result':r[4]} for r in rows])

@app.route('/delete/<int:id>', methods=['POST'])
def delete(id):
    cur = conn.cursor()
    cur.execute("DELETE FROM predictions WHERE id = %s", (id,))
    conn.commit()
    cur.close()
    return redirect(url_for('index'))

if __name__ == '__main__':
    app.run(debug=True)
