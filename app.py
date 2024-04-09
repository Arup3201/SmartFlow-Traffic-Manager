from flask import Flask, render_template, request, session, flash, redirect, g, Response
from werkzeug.security import check_password_hash, generate_password_hash
import functools
import os
from db import get_db


import os
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

from deep_sort_pytorch.deep_sort import DeepSort
from deep_sort_pytorch.utils.parser import get_config

def initialize_deepsort():
    # create the DeepSort configuration object and load settings from YAML file
    cfg_deep = get_config()
    cfg_deep.merge_from_file("deep_sort_pytorch/configs/deep_sort.yaml")

    # initialize the deep sort tracker
    deepsort = DeepSort(cfg_deep.DEEPSORT.REID_CKPT, 
                        max_dist=cfg_deep.DEEPSORT.MAX_DIST, 
                        min_confidence=cfg_deep.DEEPSORT.MIN_CONFIDENCE, 
                        nms_max_overlap=cfg_deep.DEEPSORT.NMS_MAX_OVERLAP, 
                        max_iou_distance=cfg_deep.DEEPSORT.MAX_IOU_DISTANCE, 
                        max_age=cfg_deep.DEEPSORT.MAX_AGE, 
                        n_init=cfg_deep.DEEPSORT.N_INIT, 
                        nn_budget=cfg_deep.DEEPSORT.NN_BUDGET, 
                        use_cuda=True)
    return deepsort

deepsort = initialize_deepsort()

def class_name(i):
    coco_class_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'mbrella', 'handbag', 'tie', 'sitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'srfboard', 'tennis racket', 'bottle', 'wine glass', 'cp', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli',  'carrot', 'hot dog', 'pizza', 'dont', 'cake', 'chair', 'coch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mose', 'remote', 'keyboard', 'cell phone', 'microwave',  'oven',  'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrsh']

    return coco_class_names[i]

def color_label(label):
    color = None

    if label == 0: # person
        color = (240, 62, 62) # #f03e3e
    elif label == 1: # bicycle
        color = (66, 99, 235) # #4263eb
    elif label == 2: # car
        color = (214, 51, 108) # #d6336c
    elif label == 3: # motorcycle
        color = (174, 62, 201) # #ae3ec9
    elif label == 5: # bus
        color = (112, 72, 232) # #7048e8
    elif label == 7: # truck
        color = (16, 152, 173) # #1098ad

    return color

def draw_bounding_boxes(frame, bbox_xyxys, identities, categories):
    tracked_frame = frame.copy()

    for i, xyxy in enumerate(bbox_xyxys):
        x1, y1, x2, y2 = [int(i) for i in xyxy]

        cat = int(categories[i])
        id = int(identities[i])
        color = color_label(cat)

        # If the object is not the ones we are concerned about just ignore
        if not color:
            continue
        
        cv2.rectangle(tracked_frame, (x1, y1), (x2, y2), color, 4)

        name = class_name(cat)
        text = f"{id}:{name}"
        text_size = cv2.getTextSize(text, 0, fontScale=0.5, thickness=2)[0]
        c2 = x1 + text_size[0], y1 + text_size[1] + 3
        cv2.rectangle(tracked_frame, (x1, y1), (x2, y1 + text_size[1] + 3), color, -1)
        cv2.putText(tracked_frame, text, (x1, y1 + text_size[1] + 3), 0, 0.5, color=[255, 255, 255], thickness=2, lineType=cv2.LINE_AA)

    return tracked_frame

@smart_inference_mode()
def run_tracking(
        weights=ROOT / 'yolov9-e.pt',  # model path or triton URL
        source=ROOT / 'data/images',  # file/dir/URL/glob/screen/0(webcam)
        data=ROOT / 'data/coco.yaml',  # dataset.yaml path
        imgsz=(640, 640),  # inference size (height, width)
        conf_thres=0.25,  # confidence threshold
        iou_thres=0.45,  # NMS IOU threshold
        max_det=1000,  # maximum detections per image
        device='',  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        view_img=False,  # show results
        save_txt=False,  # save results to *.txt
        save_conf=False,  # save confidences in --save-txt labels
        save_crop=False,  # save cropped prediction boxes
        nosave=False,  # do not save images/videos
        classes=None,  # filter by class: --class 0, or --class 0 2 3
        agnostic_nms=False,  # class-agnostic NMS
        augment=False,  # augmented inference
        visualize=False,  # visualize features
        update=False,  # update all models
        project=ROOT / 'runs/detect',  # save results to project/name
        name='exp',  # save results to project/name
        exist_ok=False,  # existing project/name ok, do not increment
        line_thickness=3,  # bounding box thickness (pixels)
        hide_labels=False,  # hide labels
        hide_conf=False,  # hide confidences
        half=False,  # use FP16 half-precision inference
        dnn=False,  # use OpenCV DNN for ONNX inference
        vid_stride=1,  # video frame-rate stride
        show_on_web=False, # set to True if you want tracked frames to show on websites
        save_to_db=False, # set to True if you want to get the tracking data and save it to database
):
    source = str(source)
    # save_img = not nosave and not source.endswith('.txt')  # save inference images
    save_img = False
    is_file = Path(source).suffix[1:] in (IMG_FORMATS + VID_FORMATS)
    is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
    webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
    screenshot = source.lower().startswith('screen')
    if is_url and is_file:
        source = check_file(source)  # download

    # Directories
    save_dir = increment_path(Path(project) / name, exist_ok=exist_ok)  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, dnn=dnn, data=data, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size
    if webcam:
        # view_img = check_imshow(warn=True)
        view_img = False
        dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        bs = len(dataset)
    elif screenshot:
        dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
    else:
        dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
    vid_path, vid_writer = [None] * bs, [None] * bs

    # Run inference
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
    seen, _, dt = 0, [], (Profile(), Profile(), Profile())
    for path, im, im0s, _, s in dataset:
        with dt[0]:
            im = torch.from_numpy(im).to(model.device)
            im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
            im /= 255  # 0 - 255 to 0.0 - 1.0
            if len(im.shape) == 3:
                im = im[None]  # expand for batch dim

        # Inference
        with dt[1]:
            visualize = increment_path(save_dir / Path(path).stem, mkdir=True) if visualize else False
            pred = model(im, augment=augment, visualize=visualize)
            pred = pred[0][1]

        # NMS
        with dt[2]:
            pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)

        # Second-stage classifier (optional)
        # pred = utils.general.apply_classifier(pred, classifier_model, im, im0s)

        # Process predictions
        for i, det in enumerate(pred):  # per image
            seen += 1
            if webcam:  # batch_size >= 1
                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                s += f'{i}: '
            else:
                p, im0, frame = path, im0s.copy(), getattr(dataset, 'frame', 0)

            p = Path(p)  # to Path
            s += '%gx%g ' % im.shape[2:]  # print string
            
            ims = im0.copy()
            trackings = {'trackings': []}
            tracked_frame = ims.copy()

            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()

                # Print results
                for c in det[:, 5].unique():
                    n = (det[:, 5] == c).sum()  # detections per class
                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                xywh_boxes = []
                confs = []
                oids = []
                outputs = []

                for *xyxy, conf, cls in reversed(det):
                    x1, y1, x2, y2 = xyxy
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)

                    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

                    width, height = abs(x2 - x1), abs(y2 - y1)

                    cxcywh = [cx, cy, width, height]

                    xywh_boxes.append(cxcywh)
                    confs.append(conf)
                    class_name_int = int(cls)
                    oids.append(class_name_int)

                xywhs = torch.tensor(xywh_boxes)
                confs = torch.tensor(confs)
                oids = torch.tensor(oids)

                outputs = deepsort.update(xywhs, confs, oids, ims)

                if len(outputs)>0:
                    xyxys = outputs[:, :4]
                    identities = outputs[:, -2]
                    object_ids = outputs[:, -1]
                    
                    if show_on_web:
                        tracked_frame = draw_bounding_boxes(ims, xyxys, identities, object_ids)
                        yield tracked_frame

                    if save_to_db:
                        for i, bbox_xyxy in enumerate(xyxys):
                            x1, y1, x2, y2 = [int(i) for i in bbox_xyxy]
                            width = abs(x1 - x2)
                            height = abs(y1 - y2)
                            cat = class_name(int(object_ids[i]))
                            id = int(identities[i])

                            trackings['trackings'].append({'id': id, 'class': cat, 'bounding_box': [x1, y1, width, height]})

                        yield trackings
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
    
    for frame in run_tracking(source=f"https://youtube.com/watch?v={camera['yt_id']}", show_on_web=True):
        _, frame = cv2.imencode('.jpg', frame)
        frame = frame.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

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