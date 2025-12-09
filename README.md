# Vehicle Detection and Tracker
A vehicle Detection and Tracker simply detect and track detected vehicles  in a video.  It does by detecting and assigning a a unique ID each object. 
In this project we implemented the vehicle tracker using a YOLO  model for the detection which is bases in regional propasal network while Bytetrack does the tracking.

The tracker was then teste on a test video and track all the object sucessfyly  but perform woefunly when tested on u with locally made vehicles ( Danfo and koroper ). This could be as a result of few reasons: The video taken with a low resolution camera or perhaps the the detection model was not trained on this type of distribution, hence making it difficult for the detectot to detctect. 

#  TO DO
In the future we will be training the model on a datasets that is rich with the local context so that it understand some local information which the model is not aware of. 

