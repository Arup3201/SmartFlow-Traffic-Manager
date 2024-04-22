from flask import Flask, render_template, request, session, flash, redirect, g, Response
from werkzeug.security import check_password_hash, generate_password_hash
import functools
import os
from db import get_db
from ultralytics import YOLO
import pafy
import cv2

# Flask Application
app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'sftm_db.sqlite')
    )

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

def gen_tracking_frame():
    global camera
    
    video_url = pafy.new(f"https://youtube.com/watch?v={camera['yt_id']}").getbest(preftype="mp4").url
    cap = cv2.VideoCapture(video_url)
    assert cap.isOpened(), f"Failed to open {video_url}"

    model = YOLO('yolov8m.pt')

    while True:
        _, frame = cap.read()
        results = model.track(source=frame, tracker="bytetrack.yaml", classes=[0, 1, 2, 3, 5, 7])
        annotated_frame = results[0].plot()
        
        _, tracked_frame = cv2.imencode('.jpg', annotated_frame)
        tracked_frame = tracked_frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + tracked_frame + b'\r\n')

@app.route('/object_tracking')
@login_required
def object_tracking():
    return Response(gen_tracking_frame(), mimetype="multipart/x-mixed-replace; boundary=frame")

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

if __name__=="__main__":
    app.run(host="0.0.0.0", port="5000", debug=True)