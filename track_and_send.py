from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image

from torch.multiprocessing import Queue, Process, set_start_method

import time
import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk



def counting_model(vid_queue):
    logo_detector = Detector()
    logo_detector.load_model('./weights/best.pt',img_size=640)
    print('Loaded logo detector model')
    tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=logo_detector)
    print('Loaded tracker')

    tracker.track_queue(vid_queue=vid_queue)


COCO_TRAIN = 6

if __name__ == '__main__':
    set_start_method('spawn')
    vid_queue = Queue()
    train_detector = Detector(classes=[COCO_TRAIN])
    train_detector.load_model('./yolov7-tiny.pt',img_size=320)
    print('Loaded train detector model')
    
    
    p = Process(target=counting_model, args=(vid_queue,))
    p.start()
    time.sleep(120)
    train_detector.detect_and_enqueue(0, vid_queue = vid_queue, show_live = True)
    p.join(timeout=10)