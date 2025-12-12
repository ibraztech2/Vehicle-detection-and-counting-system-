# CAPSTONE PROJECT



## 1. Abstract
In this project we implemented Vehicle Tracker which works in two stages, Detector network and the Tracker module. The Detector is from YOLO class,  a Single Shot Detector (or Single Stage Detector) [1]. The detector works by taking an image and resizing it to 448x448 [4] which is then fed into the Network. The output is then further passed to ByteTrack for assigning a unique ID [3]. This is the basics of Smart Traffic Control.

## 2.  Introduction
In Nigeria, the predominant model of transportation is by road [5].  The heavy dependence on roads has led to secondary effects like smoke emission, traffic congestion, and road accidents in some cases. This heavy usage of the road thus requires infrastructure and facilities to provide hassle-free and safe commuting.
 Nigeria is plagued by corruption and officers responsible for  road safety are not exempted from this clause. This thus stresses the need for automated solutions such as  traffic violation and infractions detection and enforcement in the Transportation sector. Artificial Intelligence  helps real time traffic analysis providing  valuable information for policymakers and road users.
In this project we introduce a vehicle detector and tracker, capable of identifying and tracking each moving vehicle. It can work in both real time and batch processing processes. This project has however been limited to detecting cars.

## 3. System Requirements
### Hardware
Laptop/PC with GPU for real-time processing
 High-Resolution Camera
Software
Python 2.9 or 3.x
 Libraries: OpenCV, NumPy, Yolov5n, ByteTrack

### 4. System Architecture
<img width="1599" height="778" alt="image" src="https://github.com/user-attachments/assets/26a24943-67a0-4bc7-890f-ec4cbb740b4c" />


### 5. Technical Implementation
The implementation is in two phases:
#### A). Detection:
We  used yolov5n for Object detection [2]. Yolo5n is a Single Shot Detector which detects and predicts a bounding box in a single stage with accuracy [1]. It takes an image and resizes it to 448x448. The image is divided into a 7x7 grid [] with each cell tasked with detection of the center of an object [1]. 
Each cell   predicts:
1. Bounding box parameters ( x, y, w, h ) 
2. Prediction score
3. Class Probabilities
4. Non-maximum Suppression ( NMS) is  applied to remove overlapping detection below a given threshold While retaining prediction above defined threshold
#### B). Tracking:
ByteTrack was implemented for the tracking [3]. It has two main components which are 
Kalman filter  for prediction of smooth object trajectories
 Hungarian Algorithm for data association.This combination thus ensure vehicles are  tracked with unique ID. 
#### C). Integration:
I. Entry Point: 
The Argparse function lets users add a command line. arguments The required argument is the path to the test video , otherwise a default video is used. 

Ii.  Video Reader: 
This function reads from a live camera or recorded video into frames and write the frames back into video with the appropriate Fourcc codec. 

Iii.  Predict function:
 Takes in each frame and returns a prediction container which houses prediction details like: bounding box, confidence, predicted label among others. It then returns a tracking list which contains all bounding box information, predicted label, confidence score.

 Iv Tracker:
 The tracking during instantiation requires the following arguments
1. Tracker buffer which is the frame rate at which object is tracked 
2. match thresh  : threshold for successful association 
3. track thresh: Threshold for an object to be tracked 
4. Mot20 : takes in bool to specify the video contains crowded scene 

The tracker update function expects detection in this format (x1, y1, x2, y2, confidence score) and returns a tracking container which contains a bounding box of tracked objects in each frame, a unique ID for each object, and tracking probabilities.

 V. Annotate Function:
 Takes in tracking results per frame and draws a bounding box, ID and class labels on it.

 Vi. Detection to json: 
Stores  detail of tracking and detections.

Code Structure:
- /Vehicle Tracker
- | — Bytetrack        (Tracker package)
- | — Checkpoints      (Yolo checkpoint)
- | — Datasets         (Test Video)
- | — Output           (Output video)
- | — main.py          (Run the detection and tracking)
- | — readme.md        (Description about the project)
- | — requirement.txts (Dependencies)


 ## 6. Results:
### a).  Inference on video 
<img width="1572" height="730" alt="image" src="https://github.com/user-attachments/assets/e0d3c806-7671-447b-89fa-98c4cadc01d6" />



### b).Inference On local Video:
![Uploading image.png…]()
![Uploading image.png…]()





## 7. Limitations:
The focus in the project is detecting cars only. In an unconstrained real world, the model failed to detect cars accurately. Even locally made  buses were  classified either  as car or train. This is because YOLO was trained on  COCO datasets [6] which do not represent Nigeria context ( our Unique Environment) nor contain objects that are made in Nigeria.  These vehicles and environment are said to fall outside the model distribution

## 8. Future Improvement:
The  inability of the model to detect our  locally made bus  shows the model lacks basic understanding of our local context, thus providing an opportunity for research work in creating Africa custom dataset upon which models are trained on. This would help AI models better interpret unique vehicles types, road condition and our unique landscape. 

## 9. References:
- [1] J. Redmon and A. Farhadi, “You Only Look Once: Unified, Real-Time Object Detection,” arXiv preprint arXiv:1506.02640, 2016.
- [2] G. Jocher, “YOLOv5 Documentation,” Ultralytics, 2020. [Online]. Available: https://github.com/ultralytics/yolov5
- [3] Y. Zhang et al., “ByteTrack: Multi-Object Tracking by Associating Every Detection Box,” arXiv preprint arXiv:2110.06864, 2021.
- [4] Roboflow, “YOLO Object Detection Explained,” 2021. [Online]. Available: https://blog.roboflow.com/yolo-object-detection
- [5] Nigerian Bureau of Statistics, “Transport Statistics Report,” 2023. [Online]. Available: https://www.nigerianstat.gov.ng
- [6] COCO Dataset, “Common Objects in Context,” 2014. [Online]. Available: https://cocodataset.org


