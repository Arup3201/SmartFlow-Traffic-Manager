from flask import Flask, render_template, request, session, flash, redirect, g
import sqlite3
import click
from werkzeug.security import check_password_hash, generate_password_hash
import functools
import os
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import pafy
import cv2
import numpy as np
from time import time
from ultralytics.utils.plotting import Annotator
from threading import Thread

# Tracking, Speed Estimation
MODEL_PATH = 'yolov8n.pt'

# Find whether a point is at right or left of a line
def find_direction_wrt_line(x, y, p):
    xp_vector = (p[0]-x[0], p[1]-x[1])
    xy_vector = (y[0]-x[0], y[1]-p[1])

    cross_product = (xp_vector[0] * xy_vector[1]) - (xp_vector[1] * xy_vector[0])

    if cross_product > 0:
        direction = -1 # left
    elif cross_product < 0:
        direction = 1 # right

    return direction

# Build a customer speed estimator to get speed for all objects inside the 4 point region
class CustomSpeedEstimator(speed_estimation.SpeedEstimator):
    def __init__(self):
        super().__init__()
        self.tracking_objs = []

    def calculate_speed(self, trk_id, track, obj_cls):
        """
        Calculation of object speed.

        Args:
            trk_id (int): object track id.
            track (list): tracking history for tracks path drawing
        """

        # Left to AB, BC, CD, DA vector
        if find_direction_wrt_line(self.reg_pts[0], self.reg_pts[1], track[-1]) < 0 and find_direction_wrt_line(self.reg_pts[1], self.reg_pts[2], track[-1]) < 0 and find_direction_wrt_line(self.reg_pts[2], self.reg_pts[3], track[-1]) < 0 and find_direction_wrt_line(self.reg_pts[3], self.reg_pts[0], track[-1]) < 0:
            direction = "known"
        else:
            direction = "unknown"

        if self.trk_previous_times[trk_id] != 0 and direction != "unknown" and trk_id not in self.trk_idslist:
            self.trk_idslist.append(trk_id)

            time_difference = time() - self.trk_previous_times[trk_id]
            if time_difference > 0:
                dist_difference = np.abs(track[-1][1] - self.trk_previous_points[trk_id][1])
                speed = dist_difference / time_difference
                self.dist_data[trk_id] = speed
                self.tracking_objs.append({'id': trk_id, 'class': obj_cls, 'speed': speed})
                

        self.trk_previous_times[trk_id] = time()
        self.trk_previous_points[trk_id] = track[-1]

    def estimate_speed(self, im0, tracks, region_color=(255, 0, 0)):
        """
        Calculate object based on tracking data.

        Args:
            im0 (nd array): Image
            tracks (list): List of tracks obtained from the object tracking process.
            region_color (tuple): Color to use when drawing regions.
        """
        self.im0 = im0
        if tracks[0].boxes.id is None:
            if self.view_img and self.env_check:
                self.display_frames()
            return im0
        self.extract_tracks(tracks)

        self.annotator = Annotator(self.im0, line_width=2)
        self.annotator.draw_region(reg_pts=self.reg_pts, color=region_color, thickness=self.region_thickness)

        for box, trk_id, cls in zip(self.boxes, self.trk_ids, self.clss):
            track = self.store_track_info(trk_id, box)

            if trk_id not in self.trk_previous_times:
                self.trk_previous_times[trk_id] = 0

            self.plot_box_and_track(trk_id, box, cls, track)
            self.calculate_speed(trk_id, track, cls)

        if self.view_img and self.env_check:
            self.display_frames()

        return im0



def set_traffic_info():
    # {class_name: [count, average_speed]}
    return {'person': [0, 0], 'car': [0, 0], 'bicycle': [0, 0], 'bus': [0, 0], 'motorcycle': [0, 0], 'truck': [0, 0]}

def save_traffic_information(url, show=False):
    # Get the model
    model = YOLO(MODEL_PATH)

    # Open the video using pafy and OpenCV
    video_url = pafy.new(url).getbest(preftype="mp4").url
    cap = cv2.VideoCapture(video_url)
    assert cap.isOpened(), f"Failed to open {video_url}"
   
    # Speed estimation
    w, h = cap.get(3), cap.get(4)
    region_pts = [(w*0.2, h*0.6), (w*0.25, h*0.7), (w*0.99, h*0.65), (w*0.8, h*0.55)]
    speed_obj = CustomSpeedEstimator()
    speed_obj.set_args(reg_pts=region_pts, 
                       names=model.names)

    # dictionaries for storing traffic data which has traffic count and estimated average speed
    id2class = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    traffic_info = set_traffic_info()

    # Process each frame of the video
    while True:
        _, frame = cap.read()

        # Track the objects in the video for particular classes we are interested in
        tracks = model.track(source=frame, tracker="bytetrack.yaml", classes=[0, 1, 2, 3, 5, 7], persist=True, show=False)

        result = speed_obj.estimate_speed(frame, tracks)

        # Calculate the total no of objects from each class and sum of speeds
        for tracking_obj in speed_obj.tracking_objs:
            obj_cls, obj_speed = tracking_obj['class'], tracking_obj['speed']
            traffic_info[id2class[int(obj_cls)]][0] += 1
            traffic_info[id2class[int(obj_cls)]][1] += obj_speed
        
        # Calculate the average speed
        for key, value in traffic_info.items():
            if value[0]:
                traffic_info[key][1] = value[1] / value[0]

        # Save the traffic information in the database
        if not show:
            pass

        # Reset the traffic volume for next iteration
        traffic_info = set_traffic_info()

        if show:
            yield result

# Flask Application
app = Flask(__name__)
app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'sftm_db.sqlite')
    )

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
            return redirect('/landing_page')

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

def login_required(view):
    @functools.wraps(view)
    def wrapped_view(**kwargs):
        if g.user is None:
            return redirect('/login')

        return view(**kwargs)
    return wrapped_view

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

@app.route('/logout')
def logout():
    session.clear()
    return redirect('/')

# Database code
def get_db():
    if 'db' not in g:
        g.db = sqlite3.connect(
            app.config['DATABASE'], 
            detect_types=sqlite3.PARSE_DECLTYPES
        )
        g.db.row_factory = sqlite3.Row

    return g.db

def close_db(e=None):
    with app.app_context():
        db = g.pop('db', None)

        if db is not None:
            db.close()

def init_db():
    with app.app_context():
        db = get_db()

        with app.open_resource('schema.sql') as f:
            db.executescript(f.read().decode('utf8'))

def clear_initialize_db():
    """Clear the existing data and create new tables."""
    init_db()
    click.echo('Initialized the database.')


if __name__=="__main__":    
    if not os.path.exists(app.config['DATABASE']):
        close_db()
        clear_initialize_db()
    else:
        q = input("Do you want to delete all previous records and start from new?")
        if q.lower() == 'yes':
            close_db()
            clear_initialize_db()

    # Start the object monitoring and saving
    thread = Thread(target=save_traffic_information, args=['https://www.youtube.com/watch?v=F5Q5ViU8QR0'], daemon=True)
    thread.start()

    app.run(host="0.0.0.0", port="5000", debug=True)