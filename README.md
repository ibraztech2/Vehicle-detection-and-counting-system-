# Vehicle Detection and Tracker
A vehicle Detection and Tracker simply detect and track detected vehicles  in a video. The project is implelmented in two stages which are detction and Tracking. The Detector detect object while  the traker keeps tracks of the detections accross frames.
  [yolov5n](https://github.com/ultralytics/yolov5) from YOLO  model was chosen for the for Object detection due to its ability to predict accurately object at a fast speed. It is based on [Single shot Detector Network (SSD)]()  Bytetrack was used to keep track of each Vehicles by assigning unique ID to detected items.
  It is important to note that, while the detetcor did very well detecting all the cars it was not able to detect the locally made vehicle. So  future work would be to make it understand local context.
  <img width="313" height="294" alt="image" src="https://github.com/user-attachments/assets/061f99e4-1b1d-49f7-ae74-3da4b096f315" />


### Libraies :
- `Opencv` : Reading video into frames and writing frames into video
- `Yolo` : handle Detetction task
- `Bytetrack` : handles Tracking


To run infernce, The repository can be forked down and then the path to video both test video and output video is passed as argument from the terminal

#  TO DO
In the future we will be training the model on a datasets that is rich with the local context so that it understand some local information which the model is not aware of. 


