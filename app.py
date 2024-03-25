from flask import Flask, render_template, request, session, flash, redirect, g
from werkzeug.security import check_password_hash, generate_password_hash
from flask_socketio import SocketIO
from db import get_db
import os
import functools
from monitoring import monitor

app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'sftm_db.sqlite')
    )

socketio = SocketIO(app, cors_allowed_origins="*")

camera = None

@app.route('/')
def index():
    return render_template("base.html")

@app.route('/register', methods=('GET', 'POST'))
def register():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None

        if not username:
            error = 'Username is required.'
        elif not password:
            error = 'Password is required.'

        if error is None:
            try:
                db.execute(
                    "INSERT INTO user (username, password) VALUES (?, ?)",
                    (username, generate_password_hash(password)),
                )
                db.commit()
            except db.IntegrityError:
                error = f"User {username} is already registered."
            else:
                return render_template('login.html')

        flash(error)

    return render_template('register.html')

@app.route('/login', methods=('GET', 'POST'))
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        db = get_db()
        error = None
        user = db.execute(
            'SELECT * FROM user WHERE username = ?', (username,)
        ).fetchone()

        if user is None:
            error = 'Incorrect username.'
        elif not check_password_hash(user['password'], password):
            error = 'Incorrect password.'

        if error is None:
            session.clear()
            session['user_id'] = user['id']
            return redirect('/location')

        flash(error)

    return render_template('login.html')

@app.before_request
def load_logged_in_user():
    user_id = session.get('user_id')

    if user_id is None:
        g.user = None
    else:
        g.user = get_db().execute(
            'SELECT * FROM user WHERE id = ?', (user_id,)
        ).fetchone()

    camera_id = session.get('camera_id')
    
    if camera_id is None:
        g.camera = None
    else:
        g.camera = get_db().execute(
            'SELECT * FROM camera WHERE c_id=?', (camera_id, )
        ).fetchone()
        
    global camera
    camera = g.camera

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect('/login')

        return view(**kwargs)
    return wrapped_view

@app.route('/location')
@login_required
def location():
    return render_template('location.html')

@app.route('/landing_page', methods=('GET', 'POST'))
@login_required
def landing_page():
    if request.method == "POST":
        loc = request.form['choice']
        camera = get_db().execute(
            'SELECT * FROM camera WHERE c_name = ?', (loc,)
        ).fetchone()
        
        session['camera_id'] = camera['c_id']

    return render_template('landing.html')

@app.route('/dashboard')
def dashboard():
    return render_template('dashboard.html')

@socketio.on("connect")
def connect():
    socketio.start_background_task(target=gen_monitoring_data)

def gen_monitoring_data():
    global camera
    
    for traffic_data in monitor('model_data/best.pt', f"https://youtube.com/watch?v={camera['yt_id']}"):
        
        socketio.emit("data", {"response": traffic_data})

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__=="__main__":
    socketio.run(app, host="0.0.0.0", port="5000", debug=True)