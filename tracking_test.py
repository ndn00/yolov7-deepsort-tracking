from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk

detector = Detector() 
print("Initialized Detector")
detector.load_model('./weights/best.pt',) # pass the path to the trained weight file
print('Loaded model')

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)
print('Initialized Tracker')
# output = None will not save the output video
# tracker.track_video(0, output="./IO_data/output/output.avi", outcsv="./IO_data/output/outputcsv" show_live = True, skip_frames = 0, count_objects = True, verbose=1)
tracker.track_video("./IO_data/input/video/testvid_clear.mp4", output="./IO_data/output/output.avi", outcsv="./IO_data/output/output.csv", show_live = True, skip_frames = 0, count_objects = True, verbose=1)
print('Output video')