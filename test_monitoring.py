from ultralytics import YOLO
from ultralytics.solutions import speed_estimation
import cv2 as cv
import pafy
import numpy as np

import time
from ultralytics.utils.plotting import Annotator

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

        if not ((self.reg_pts[0][0] < track[-1][0] < self.reg_pts[1][0]) and(self.reg_pts[2][0] < track[-1][0] < self.reg_pts[3][0])):
            return
        
        if self.reg_pts[3][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[3][1] + self.spdl_dist_thresh:
            direction = "known"
        elif self.reg_pts[2][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[2][1] + self.spdl_dist_thresh:
            direction = "known"
        elif self.reg_pts[1][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[1][1] + self.spdl_dist_thresh:
            direction = "known"
        elif self.reg_pts[0][1] - self.spdl_dist_thresh < track[-1][1] < self.reg_pts[0][1] + self.spdl_dist_thresh:
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
    return {'person': [0, 0], 'car': [0, 0], 'bicycle': [0, 0], 'bus': [0, 0], 'motorcycle': [0, 0], 'truck': [0, 0]}

def main():
    # Get the model
    model = YOLO('yolov8m.pt')

    # Open the video using pafy and OpenCV
    url = "https://www.youtube.com/watch?v=F5Q5ViU8QR0"
    video_url = pafy.new(url).getbest(preftype="mp4").url
    cap = cv.VideoCapture(video_url)
    assert cap.isOpened(), f"Failed to open {video_url}"
   
    # Speed estimation
    w, h = cap.get(3), cap.get(4)
    print(w, h)
    region_pts = [(w*0.2, h*0.6), (w*0.25, h*0.7), (w*0.99, h*0.65), (w*0.8, h*0.55)]
    # region_pts = [(w*0.2, h*0.6), (w*0.25, h*0.7)]
    speed_obj = CustomSpeedEstimator()
    speed_obj.set_args(reg_pts=region_pts, 
                       names=model.names, 
                       view_img=True)

    # dictionaries for storing traffic data which has traffic count and estimated average speed
    id2class = {0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}
    traffic_info = set_traffic_info()

    # Process each frame of the video
    while True:
        _, frame = cap.read()

        # Track the objects in the video for particular classes we are interested in
        tracks = model.track(source=frame, tracker="bytetrack.yaml", classes=[0, 1, 2, 3, 5, 7], persist=True, show=False)

        speed_obj.estimate_speed(frame, tracks)

        # Calculate the total no of objects from each class and sum of speeds
        for tracking_obj in speed_obj.tracking_objs:
            _, obj_cls, obj_speed = tracking_obj

            traffic_info[id2class[int(obj_cls)]][0] += 1
            traffic_info[id2class[int(obj_cls)]][1] += obj_speed
        
        # Calculate the average speed
        for key, value in traffic_info.items():
            traffic_info[key][1] = value[1] / value[0]

        print(traffic_info)

        # Reset the traffic volume for next iteration
        traffic_info = set_traffic_info()

        # Quitting conditions
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

if __name__=="__main__":
    main()