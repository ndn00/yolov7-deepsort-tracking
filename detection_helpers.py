
import cv2
import torch
from numpy import random
import numpy as np
import time

from models.experimental import attempt_load
from utils.datasets import letterbox, np
from utils.general import check_img_size, non_max_suppression, apply_classifier,scale_coords, xyxy2xywh
from utils.plots import plot_one_box
from utils.torch_utils import select_device, load_classifier,TracedModel

from torch.multiprocessing import Queue

class Detector:
    def __init__(self, conf_thres:float = 0.25, iou_thresh:float = 0.45, agnostic_nms:bool = False, save_conf:bool = False, classes:list = None):
        '''
        args:
        conf_thres: Thresholf for Classification
        iou_thres: Thresholf for IOU box to consider
        agnostic_nms: whether to use Class-Agnostic NMS
        save_conf: whether to save confidences in 'save_txt' labels afters inference
        classes: Filter by class from COCO. can be in the format [0] or [0,1,2] etc
        '''
        self.device = select_device("0" if torch.cuda.is_available() else 'cpu')
        self.conf_thres = conf_thres
        self.iou_thres = iou_thresh
        self.classes = classes
        self.agnostic_nms = agnostic_nms
        self.save_conf = save_conf


    def load_model(self, weights:str, img_size:int = 640, trace:bool = True, classify:bool = False):
        '''
        weights: Path to the model
        img_size: Input image size of the model
        trace: Whether to trace the model or not
        classify: whether to load the second stage classifier model or not
        '''
        self.half = self.device.type != 'cpu'  # half precision only supported on CUDA

        self.model = attempt_load(weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.imgsz = check_img_size(img_size, s=self.stride)  # check img_size

        if trace:
            self.model = TracedModel(self.model, self.device, img_size)

        if self.half:
            self.model.half()  # to FP1
        
        # Run inference for CUDA just once
        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, self.imgsz, self.imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

        # Second-stage classifier
        self.classify = classify
        if classify:
            self.modelc = load_classifier(name='resnet101', n=2)  # initialize
            self.modelc.load_state_dict(torch.load('weights/resnet101.pt', map_location=self.device)['model']).to(self.device).eval()

         # Get names and colors of Colors for BB creation
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.names]
        


    @torch.no_grad()
    def detect(self, source, plot_bb:bool =True):
        '''
        source: Path to image file, video file, link or text etc
        plot_bb: whether to plot the bounding box around image or return the prediction
        '''
        img, im0 = self.load_image(source)
        img = torch.from_numpy(img).to(self.device)
        img = img.half() if self.half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3: # Single batch -> single image
            img = img.unsqueeze(0)

        # Inference
        pred = self.model(img, augment=False)[0] # We don not need any augment during inference time

        # Apply NMS
        pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, classes=self.classes, agnostic=self.agnostic_nms)

        # Apply Classifier
        if self.classify:
            pred = apply_classifier(pred, self.modelc, img, im0) # I thnk we need to add a new axis to im0


        # Post - Process detections
        det = pred[0]# detections per image but as we have  just 1 image, it is the 0th index
        if len(det):
            # Rescale boxes from img_size to im0 size
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

            # Write results
            for *xyxy, conf, cls in reversed(det):

                if plot_bb:  # Add bbox to image   # save_img
                    label = f'{self.names[int(cls)]} {conf:.2f}'
                    plot_one_box(xyxy, im0, label=label, color=self.colors[int(cls)], line_thickness=1)
                    
        
            return im0 if plot_bb else det.detach().cpu().numpy()

        return im0 if plot_bb else None # just in case there's no detection, return the original image. For tracking purpose plot_bb has to be False always
        
    def detect_and_save(self,video:str, output_dir:str, detect_frames:int=30, num_failed_detect_allowed:int=5, show_live:bool=False, verbose:int = 1):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output_dir: path to output directory
            detect_frames: run detection every nth frames 
            num_failed_detect_allowed: consecutive failed detections allowed before stop recording
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        num_failed_detect = 0
        frame_num = 0
        out = None
        yolo_dets = None

        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            if detect_frames and not frame_num % detect_frames:  
                yolo_dets = self.detect(frame.copy(), plot_bb = False)
            else:
                yolo_dets = yolo_dets  # Get the detections
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if yolo_dets is not None:
                num_failed_detect = 0
                if out == None:
                    output = output_dir+"/"+str(time.time())+"train.avi"
                    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
                    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    fps = int(vid.get(cv2.CAP_PROP_FPS))
                    codec = cv2.VideoWriter_fourcc(*"XVID")
                    out = cv2.VideoWriter(output, codec, fps, (width, height))
            elif out != None:
                num_failed_detect+=1
                if num_failed_detect==num_failed_detect_allowed:
                    num_failed_detect = 0
                    out = None
            if out:
                out.write(result)
                cv2.putText(result, "Recording to: {}".format(out), (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
            fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
            print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()

    def detect_and_enqueue(self,video:str, vid_queue:Queue, detect_frames:int=60, num_failed_detect_allowed:int=10, show_live:bool=False, verbose:int = 1):
        '''
        Track any given webcam or video
        args: 
            video: path to input video or set to 0 for webcam
            output_dir: path to output directory
            detect_frames: run detection every nth frames 
            num_failed_detect_allowed: consecutive failed detections allowed before stop recording
            show_live: Whether to show live video tracking. Press the key 'q' to quit
            verbose: print details on the screen allowed values 0,1,2
        '''
        try: # begin video capture
            vid = cv2.VideoCapture(int(video))
        except:
            vid = cv2.VideoCapture(video)

        num_failed_detect = 0
        frame_num = 0
        yolo_dets = None
        recording = False
        while True: # while video is running
            return_value, frame = vid.read()
            if not return_value:
                print('Video has ended or failed!')
                break
            frame_num +=1

            start_time = time.time()

            # -----------------------------------------PUT ANY DETECTION MODEL HERE -----------------------------------------------------------------
            if detect_frames and not frame_num % detect_frames:  
                yolo_dets = self.detect(frame.copy(), plot_bb = False)
            else:
                yolo_dets = yolo_dets  # Get the detections
            
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = np.asarray(frame)
            result = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            
            if yolo_dets is not None:
                num_failed_detect = 0
                if recording == False:
                    recording = True
            elif recording == True:
                num_failed_detect+=1
                if num_failed_detect==num_failed_detect_allowed:
                    num_failed_detect = 0
                    recording = False
                    width = int(vid.get(cv2.CAP_PROP_FRAME_WIDTH))  # by default VideoCapture returns float instead of int
                    height = int(vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_queue.put(np.zeros((width, height, 3), np.uint8)) # blank frame marks end of video
            if recording:
                vid_queue.put(result.copy())
                cv2.putText(result, "Recording to Queue", (5, 35), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 0), 2)
            fps = 1.0 / (time.time() - start_time) # calculate frames per second of running detections
            print(f"Processed frame no: {frame_num} || Current FPS: {round(fps,2)}")

            if show_live:
                cv2.imshow("Output Video", result)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

        cv2.destroyAllWindows()

    def load_image(self, img0):
        '''
        Load and pre process the image
        args: img0: Path of image or numpy image in 'BGR" format
        '''
        if isinstance(img0, str): img0 = cv2.imread(img0)  # BGR
        assert img0 is not None, 'Image Not Found '

        # Padded resize
        img = letterbox(img0, self.imgsz, stride=self.stride)[0]

        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        return img, img0
    

    def save_txt(self, det, im0_shape, txt_path):
        '''
        Save the results of an image in a .txt file
        args:
            det: detecttions from the model
            im0_shape: Shape of Original image
            txt_path: File of the text path
        '''
        gn = torch.tensor(im0_shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
            line = (cls, *xywh, conf) if self.save_conf else (cls, *xywh)  # label format
            with open(txt_path + '.txt', 'a') as f:
                f.write(('%g ' * len(line)).rstrip() % line + '\n')
