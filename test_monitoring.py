from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2 as cv
import pafy
import numpy as np

from time import time
from ultralytics.utils.plotting import Annotator

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

def main():
    # Get the model
    model = YOLO(MODEL_PATH)

    # Open the video using pafy and OpenCV
    url = "https://www.youtube.com/watch?v=F5Q5ViU8QR0"
    video_url = pafy.new(url).getbest(preftype="mp4").url
    cap = cv.VideoCapture(video_url)
    assert cap.isOpened(), f"Failed to open {video_url}"
   
    # Speed estimation
    w, h = cap.get(3), cap.get(4)
    region_pts = [(w*0.1, h*0.55), (w*0.25, h*0.8), (w*0.99, h*0.75), (w*0.99, h*0.5)]
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

        cv.imshow("Traffic Monitoring", result)
        
        print(traffic_info)
        
        # Reset the traffic volume for next iteration
        traffic_info = set_traffic_info()

        # Quitting conditions
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    main()