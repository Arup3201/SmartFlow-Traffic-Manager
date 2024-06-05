import numpy as np
import cv2
import sqlite3
from PIL import Image

def save_img(img):
    arr_np = np.frombuffer(img, np.uint8)
    img_np = cv2.imdecode(arr_np, cv2.IMREAD_COLOR)
    cv2.imwrite('sftm_server/static/img/temp.jpg', img_np)
    return 'sftm_server/static/img/temp.jpg'

def get_accident_img(acc_id):
    db = sqlite3.connect('instance/sftm.sqlite')
    db.row_factory = sqlite3.Row
    
    data = db.execute(
        'SELECT * FROM accidents WHERE acc_id=?', (acc_id, )
    ).fetchone()
    
    return data['img']

if __name__=="__main__":
    acc_id = 1
    img = get_accident_img(acc_id=acc_id)
    img_path = save_img(img)
    
    Image.open(img_path)