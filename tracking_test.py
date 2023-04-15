from detection_helpers import *
from tracking_helpers import *
from  bridge_wrapper import *
from PIL import Image

# from init_arduino import *

# import gi
# gi.require_version('Gtk', '3.0')
# from gi.repository import Gtk

detector = Detector() 
print("Initialized Detector")
detector.load_model('./weights/yolov7_traincar.pt',img_size=640, classify=True) # pass the path to the trained weight file
# detector.load_model('traced_model.pt',trace=False)
print('Loaded model')

# Initialise  class that binds detector and tracker in one class
tracker = YOLOv7_DeepSORT(reID_model_path="./deep_sort/model_weights/mars-small128.pb", detector=detector)
print('Initialized Tracker')
# output = None will not save the output video
# tracker.track_video(0, output="./IO_data/output/output.avi", outcsv="./IO_data/output/outputcsv" show_live = True, skip_frames = 0, count_objects = True, verbose=1)
count = tracker.track_video("./IO_data/input/video/direct_view.mp4", output="./IO_data/output/output.avi", show_live = True, skip_frames = 0, count_objects = True, verbose=1)
# count = tracker.track_video(0, output="./IO_data/output/output.avi", show_live = True, skip_frames = 0, count_objects = True, verbose=1)
print('Output video')
# send_count(count)