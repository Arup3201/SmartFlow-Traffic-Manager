from flask import Flask, render_template, current_app, request, session, flash, redirect, g, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
import functools
import os
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import pafy
import cv2
import numpy as np
from time import time
import datetime
from ultralytics.utils.plotting import Annotator
from threading import Thread
from .db import get_db, traffic_db, init_app

# Model path
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

def save_traffic_information(url, app_context):
    print("SAVING TRAFFIC INFORMATION INTO DATABASE ...")

    app_context.push()

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
        traffic_db(traffic_info=traffic_info)

        # Reset the traffic volume for next iteration
        traffic_info = set_traffic_info()


def create_app(test_config=None):
    # create and configure the app
    app = Flask(__name__, instance_relative_config=True)
    app.config.from_mapping(
        SECRET_KEY='dev',
        DATABASE=os.path.join(app.instance_path, 'sftm.sqlite'),
    )

    if test_config is None:
        # load the instance config, if it exists, when not testing
        app.config.from_pyfile('config.py', silent=True)
    else:
        # load the test config if passed in
        app.config.from_mapping(test_config)

    # ensure the instance folder exists
    try:
        os.makedirs(app.instance_path)
    except OSError:
        pass

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
    @login_required
    def dashboard():
        return render_template('dashboard.html')

    @app.route('/get_div_data')
    @login_required
    def get_div_data():
        data = [
            {
                'amount': 0, 
                'percentage': 0,
                'change': ''
            }, 
            {
                'amount': 0, 
                'percentage': 0,
                'change': ''
            }, 
            {
                'congestion-level': '', 
                'change': ''
            }
        ]

        db = get_db()
        today_traffic = db.execute(
            '''SELECT ROUND(AVG(pedestrian_count+car_count+bus_count+bicycle_count+motorcycle_count+truck_count)) AS avg_traffic_count, ROUND(AVG(volume)) AS avg_volume, MAX(congestion) AS overall_congestion 
            FROM traffic 
            WHERE date_time LIKE (?)''', (datetime.date.today().strftime("%Y-%m-%d")+'%', )).fetchone()
        yesterday_traffic = db.execute(
            '''SELECT ROUND(AVG(pedestrian_count+car_count+bus_count+bicycle_count+motorcycle_count+truck_count)) AS avg_traffic_count,ROUND(AVG(volume)) AS avg_volume, MAX(congestion) AS overall_congestion 
            FROM traffic 
            WHERE date_time LIKE (?)''', ((datetime.date.today()-datetime.timedelta(days=1)).strftime("%Y-%m-%d")+'%', )).fetchone()
        
        data[0]['amount'] = today_traffic['avg_volume']
        data[0]['percentage'] = round(((today_traffic['avg_volume'] - yesterday_traffic['avg_volume']) / yesterday_traffic['avg_volume']) * 100, 1)
        data[0]['change'] = 'increase' if data[0]['percentage'] > 0 else 'decrease'
        
        data[1]['amount'] = today_traffic['avg_traffic_count']
        data[1]['percentage'] = round(((today_traffic['avg_traffic_count'] - yesterday_traffic['avg_traffic_count']) / yesterday_traffic['avg_traffic_count']) * 100, 1)
        data[1]['change'] = 'increase' if data[1]['percentage'] > 0 else 'decrease'
        
        congestion_levels = ['Low', 'Medium', 'High']
        data[2]['congestion-level'] = congestion_levels[today_traffic['overall_congestion']]
        data[2]['change'] = 'Higher than yesterday' if today_traffic['overall_congestion'] > yesterday_traffic['overall_congestion'] else 'Lower or equal to yesterday'
        
        print(data)
        return data

    @app.route('/line_chart_data')
    @login_required
    def line_chart_data():
        db = get_db()
        
        series_data = [
            {
                'name': 'Pedestrian', 
                'data': []
            }, 
            {
                'name': 'Car', 
                'data': []
            }, 
            {
                'name': 'Bus', 
                'data': []
            }, 
            {
                'name': 'Bicycle', 
                'data': []
            }, 
            {
                'name': 'Motorcycle', 
                'data': []
            }, 
            {
                'name': 'Truck', 
                'data': []
            }, 
            {
                'name': 'Traffic Volume', 
                'data': []
            }
        ]
        
        for hour in range(24):
            def get_avg_in_db(string):
                data = db.execute(
                    f"SELECT ROUND(AVG({string})) AS avg_{string} FROM traffic WHERE date_time LIKE (?)", (datetime.date.today().strftime("%Y-%m-%d")+f' {hour:0>2}%', )
                ).fetchone()
                return data[f'avg_{string}'] if data[f'avg_{string}'] else 0
        
            series_data[0]['data'].append(get_avg_in_db('pedestrian_count'))
            series_data[1]['data'].append(get_avg_in_db('car_count'))
            series_data[2]['data'].append(get_avg_in_db('bus_count'))
            series_data[3]['data'].append(get_avg_in_db('bicycle_count'))
            series_data[4]['data'].append(get_avg_in_db('motorcycle_count'))
            series_data[5]['data'].append(get_avg_in_db('truck_count'))
            series_data[6]['data'].append(get_avg_in_db('volume'))
            
        categories = [f'{hour:0>2}:00' for hour in range(24)]

        print([{
            'series_data': series_data, 
            'categories': categories, 

        }])

        return jsonify([{
            'series_data': series_data, 
            'categories': categories, 

        }])

    @app.route('/angular_chart_data')
    @login_required
    def angular_chart_data():
        data = [{
                    'value': [],
                    'name': 'Average Speed'
                },
                {
                    'value': [],
                    'name': 'Total Count'
                }]
        
        db = get_db()
        
        def get_avg_in_db(string):
            data = db.execute(
                f"SELECT ROUND(AVG({string})) AS avg_{string}                 FROM traffic WHERE date_time LIKE (?)", (datetime.date.today().strftime("%Y-%m-%d")+'%', )
            ).fetchone()
            return data[f'avg_{string}'] if data[f'avg_{string}'] else 0
            
        objs = ['pedestrian', 'car', 'bus', 'bicycle', 'motorcycle', 'truck']
        for ob in objs:
            data[0]['value'].append(get_avg_in_db(f'{ob}_speed'))
            data[1]['value'].append(get_avg_in_db(f'{ob}_count'))
        
        print(data)
        return jsonify(data)

    @app.route('/pie_chart_data')
    @login_required
    def pie_chart_data():
        data = [{
                'value': 0,
                'name': 'Pedestrian'
                },
                {
                    'value': 0,
                    'name': 'Car'
                },
                {
                    'value': 0,
                    'name': 'Bus'
                },
                {
                    'value': 0,
                    'name': 'Bicycle'
                },
                {
                    'value': 0,
                    'name': 'Motorcycle'
                }, 
                {
                    'value': 0,
                    'name': 'Truck'
                }]
        
        db = get_db()
        def get_volume_in_db(string):
            data = db.execute(
                f"SELECT ROUND(AVG({string}_count) * AVG({string}_speed)) AS {string}_volume FROM traffic WHERE date_time LIKE (?)", (datetime.date.today().strftime("%Y-%m-%d")+'%', )
            ).fetchone()
            return data[f'{string}_volume'] if data[f'{string}_volume'] else 0
        
        objs = ['pedestrian', 'car', 'bus', 'bicycle', 'motorcycle', 'truck']
        for i, ob in enumerate(objs):
            data[i]['value'] = get_volume_in_db(ob)
        
        print(data)
        return jsonify(data)

    @app.route('/highest_traffic')
    def highest_traffic():
        data = []
        
        db = get_db()
        
        highest_traffic_rows = db.execute(
            '''SELECT strftime ('%H', date_time) AS hour_day, ROUND(AVG(volume), 2) as volume, ROUND(AVG(pedestrian_speed+car_speed+bus_speed+bicycle_speed+motorcycle_speed+truck_speed)) AS avg_speed, 
            CASE MAX(AVG(pedestrian_count), AVG(car_count), AVG(bus_count), AVG(bicycle_count), AVG(motorcycle_count), AVG(truck_count))
                WHEN AVG(pedestrian_count)
                    THEN 'Pedestrian'
                WHEN  AVG(car_count)
                    THEN 'Car'
                WHEN  AVG(bus_count)
                    THEN 'Bus'
                WHEN  AVG(bicycle_count)
                    THEN 'Bicycle'
                WHEN  AVG(motorcycle_count)
                    THEN 'Motorcycle'
                WHEN  AVG(truck_count)
                    THEN 'Truck'
                END highest_vehicle
            FROM traffic 
            GROUP BY strftime ('%H', date_time)
            HAVING date_time LIKE (?)
            ORDER BY volume DESC
            LIMIT 5''', (datetime.date.today().strftime("%Y-%m-%d")+'%', )).fetchall()
                
        for highest_row in highest_traffic_rows:
            data.append(
                {
                    'time': highest_row['hour_day'], 
                    'volume': highest_row['volume'], 
                    'speed': highest_row['avg_speed'], 
                    'highest-vehicle': highest_row['highest_vehicle']
                }
            )
            
        print(data)
        return data

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect('/')

    init_app(app)

    if not app.config.get('TRAFFIC_MONITORING_STARTED', False):
        app.config['TRAFFIC_MONITORING_STARTED'] = True
        # Start the object monitoring and saving on a new thread
        thread = Thread(target=save_traffic_information, args=['https://www.youtube.com/watch?v=F5Q5ViU8QR0', app.app_context()], daemon=True)
        thread.start()

    return app