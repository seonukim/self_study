from flask import Flask, render_template, request
import sqlite3
import os
path1 = '/Users/kimseonu/Desktop/self_study/self_study/data/DB/'
path2 = '/Users/kimseonu/Desktop/self_study/self_study/Tensorflow_Project/FLASK/template/'
os.chdir(path1)
os.chdir(path2)

app = Flask(__name__)

# 데이터베이스 만들기
conn = sqlite3.connect(path2 + "wanggun.db")
cursor = conn.cursor()
cursor.execute("SELECT * FROM general;")
print(cursor.fetchall())

@app.route('/')
def run():
    conn = sqlite3.connect(path2 + "wanggun.db")
    c = conn.cursor()
    c.execute("SELECT * FROM general;")
    rows = c.fetchall()
    return render_template(path1 + "board_index.html", rows=rows)

@app.route('/modi')
def modi():
    ids = request.args.get('id')
    conn = sqlite3.connect(path2 + "wanggun.db")
    c = conn.cursor()
    c.execute('SELECT * FROM general where id = ' + str(ids))
    rows = c.fetchall()
    return render_template(path1 + 'board_modi.html', rows=rows)

@app.route('/addrec', methods=['POST', 'GET'])
def addrec():
    if request.method == 'POST':
        try:
            conn = sqlite3.connect(path2 + "wanggun.db")
            war = request.form['war']
            ids = request.form['id']
            c = conn.cursor()
            c.execute('UPDATE general SET war = '+ str(war) + " WHERE id = "+str(ids))
            conn.commit()
            msg = '정상적으로 입력되었습니다.'
        except:
            conn.rollback()
            msg = '에러가 발생하였습니다.'
        finally:
            conn.close()
            return render_template(path1 + "board_result.html", msg=msg)

app.run(host='127.0.0.1', port=5000, debug=False)