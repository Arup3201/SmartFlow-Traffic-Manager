from flask import Flask, render_template, request, session, flash, redirect, g, jsonify
from werkzeug.security import check_password_hash, generate_password_hash
import functools
import os
from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import pafy
import cv2
import numpy as np
from time import time, sleep
import datetime
from ultralytics.utils.plotting import Annotator
from threading import Thread
from .db import get_db, save_to_traffic_db, save_to_accident_db, init_app

# Live video url id from youtube
# Accident Video
# LIVE_URL_ID = '3SeZbDXtTCI'
# Normal Traffic Camera
LIVE_URL_ID = 'F5Q5ViU8QR0'

# Model path
MODEL_PATH = 'yolov8n.pt'
ACCIDENT_DET_MODEL_PATH = 'accident_detection.pt'

# Dictionaries
ID2TRAFFICCLASS = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
ID2ACCIDENTCLASS = {0: 'Bike, Bike', 1: 'Bike, Object', 2: 'Bike, Pedestrian', 3: 'Car, Bike', 4:'Car, Car', 5: 'Car, Object', 6: 'Car, Pedestrian'}

# List for use
CONGESTION_LEVELS = ['Low', 'Medium', 'High']

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

def convertBinary(img):
    bin_img = cv2.resize(img, None, fx=0.75, fy=0.75)
    bin_img = cv2.imencode('.jpg', bin_img)[1].tobytes()
    return bin_img

def save_traffic_information(url, app_context):
    print(f"FETCHING TRAFFIC DATA FROM {url}")
    print("SAVING TRAFFIC INFORMATION INTO DATABASE ...")

    app_context.push()

    # Get the models
    model = YOLO(MODEL_PATH)
    accident_detector = YOLO(ACCIDENT_DET_MODEL_PATH)

    # Open the video using pafy and OpenCV
    video_url = pafy.new(url).getbest(preftype="mp4").url
    cap = cv2.VideoCapture(video_url)
    assert cap.isOpened(), f"Failed to open {video_url}"

    # Speed estimation
    w, h = cap.get(3), cap.get(4)
    region_pts = [(w*0.2, h*0.6), (w*0.25, h*0.8), (w*0.99, h*0.5), (w*0.99, h*0.75)]
    speed_obj = CustomSpeedEstimator()
    speed_obj.set_args(reg_pts=region_pts, 
                    names=model.names)
    
    traffic_info = set_traffic_info()
    
    # Whether the accident is new or not based on time
    is_new_accident = True
    last_accident_time = 0

    # Process each frame of the video
    while True:
        # Get the current video frame
        _, frame = cap.read()

        # Track the objects in the video for particular classes we are interested in
        tracks = model.track(source=frame, tracker="bytetrack.yaml", classes=[0, 1, 2, 3, 5, 7], persist=True, imgsz=640, show=False, verbose=False)

        # Estimate speed using ultralytics speed estimator
        speed_obj.estimate_speed(frame, tracks)
        
        # Detect accidents using the model for particular classes
        result = accident_detector(source=frame, imgsz=640, verbose=False)
        bboxes = result[0].boxes
        
        current_time = time() # Save the current time for checking new accidents
        
        if len(bboxes.conf) > 0:
            ind = np.argmax(bboxes.conf)
                
            # If the confidence score is more than 80% and the accident is a new accident more than 15 mins after the previous one
            if bboxes.conf[ind] > 0.8 and is_new_accident:
                print("A new accident is detected!!")
                save_to_accident_db({
                    'img': convertBinary(frame), 
                    'involved': ID2ACCIDENTCLASS[int(bboxes.cls[ind])], 
                })
                is_new_accident = False
                last_accident_time = time() # Save the time of the accident
        
        # If time difference with previous accident is more than 15 mins then the next accident should be taken into account
        if abs(current_time - last_accident_time) > 900:
            is_new_accident = True
            
        # Calculate the total no of objects from each class and sum of speeds
        for tracking_obj in speed_obj.tracking_objs:
            obj_cls, obj_speed = tracking_obj['class'], tracking_obj['speed']
            traffic_info[ID2TRAFFICCLASS[int(obj_cls)]][0] += 1
            traffic_info[ID2TRAFFICCLASS[int(obj_cls)]][1] += obj_speed
        
        # Calculate the average speed
        for key, value in traffic_info.items():
            if value[0]:
                traffic_info[key][1] = value[1] / value[0]

        # Save the traffic information in the database
        save_to_traffic_db(traffic_info=traffic_info)

        # Reset the traffic volume for next iteration
        traffic_info = set_traffic_info()
        
        # Sleep for 2 secs
        sleep(2)

def get_img_path(img):
    arr_np = np.fromstring(img, np.uint8)
    img_np = cv2.imdecode(arr_np, cv2.CV_BGR2RGB)
    cv2.imwrite('static/img/temp.jpg', img_np)
    return 'img/temp.jpg'
    

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
            g.video_id = LIVE_URL_ID

    def login_required(view):
        @functools.wraps(view)
        def wrapped_view(**kwargs):
            if g.user is None:
                return redirect('/login')

            return view(**kwargs)
        return wrapped_view

    @app.route('/landing_page')
    @login_required
    def landing_page():
        return render_template('landing.html')

    @app.route('/dashboard')
    @login_required
    def dashboard():
        return render_template('dashboard.html')

    @app.route('/traffic_analysis')
    @login_required
    def traffic_analysis():
        return render_template('traffic_analysis.html')


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
                'congestion-level': 'Low', 
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
        
        if today_traffic['avg_volume']:
            data[0]['amount'] = today_traffic['avg_volume']
            data[1]['amount'] = today_traffic['avg_traffic_count']
            data[2]['congestion-level'] = CONGESTION_LEVELS[today_traffic['overall_congestion']]
            if yesterday_traffic['avg_volume']:
                data[0]['percentage'] = round(((today_traffic['avg_volume'] - yesterday_traffic['avg_volume']) / yesterday_traffic['avg_volume']) * 100, 1)
                data[0]['change'] = 'increase' if data[0]['percentage'] > 0 else 'decrease'
                
                data[1]['percentage'] = round(((today_traffic['avg_traffic_count'] - yesterday_traffic['avg_traffic_count']) / yesterday_traffic['avg_traffic_count']) * 100, 1)
                data[1]['change'] = 'increase' if data[1]['percentage'] > 0 else 'decrease'
                
                data[2]['change'] = 'Higher than yesterday' if today_traffic['overall_congestion'] > yesterday_traffic['overall_congestion'] else 'Lower or equal to yesterday'
            else:
                data[0]['percentage'] = 100
                data[0]['change'] = 'increase'
                
                data[1]['percentage'] =100
                data[1]['change'] = 'increase'
                
                data[2]['change'] = 'Higher than yesterday'
        
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

    @app.route('/get_modal_data/<string:acc_id>')
    @login_required
    def get_modal_data(acc_id):
        modal_data = {}
        
        db = get_db()
        data = db.execute(
            "SELECT acc_id, date_time, img, involved, severity, stat FROM accidents WHERE acc_id=?", (acc_id, )
        ).fetchone()
        
        modal_data['id'] = data['acc_id']
        modal_data['time'] = data['date_time']
        modal_data['image'] = get_img_path(data['img'])
        modal_data['involved'] = data['involved']
        modal_data['severity'] = data['severity']
        modal_data['stat'] = data['stat']
        
        print(modal_data)
        return jsonify([modal_data])

    @app.route('/get_accident_data')
    @login_required
    def get_accident_data():
        accidents = []
        
        db = get_db()
        data = db.execute(
            "SELECT acc_id, date_time, involved, severity, stat FROM accidents WHERE date_time LIKE (?)", (datetime.date.today().strftime("%Y-%m-%d")+"%", )
        ).fetchall()
        
        for row in data:
            accidents.append(
                {
                    'id': row['acc_id'], 
                    'time': row['date_time'], 
                    'involved': row['involved'], 
                    'severity': row['severity'], 
                    'stat': row['stat']
                }
            )
        
        print(accidents)
        return accidents

    @app.route('/line_chart_analysis/<string:timeline>')
    @login_required
    def line_chart_analysis(timeline):
        pass
    
    @app.route('/angular_chart_analysis/<string:timeline>')
    @login_required
    def angular_chart_analysis(timeline):
        pass
    
    @app.route('/pie_chart_analysis/<string:timeline>')
    @login_required
    def pie_chart_analysis(timeline):
        pass

    @app.route('/logout')
    def logout():
        session.clear()
        return redirect('/')

    init_app(app)

    if not app.config.get('TRAFFIC_MONITORING_STARTED', False):
        app.config['TRAFFIC_MONITORING_STARTED'] = True
        # Start the object monitoring and saving on a new thread
        thread = Thread(target=save_traffic_information, args=[f"https://www.youtube.com/watch?v={LIVE_URL_ID}", app.app_context()], daemon=True)
        thread.start()

    return app