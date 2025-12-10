import os
import json
import cv2 as cv
import numpy as np
from ultralytics import YOLO
import time
import random

from bytetrack import yolox
from bytetrack.yolox.tracker.byte_tracker import BYTETracker


def video_reader(path, model):
    global tracker
    start_time = time.time()
    cap = cv.VideoCapture(path)
    map_list = ("car",)

    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate =   int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    bounding_color_dict = {0: (0, 255, 0)}

    tracker = BYTETracker(args, frame_rate=frame_rate)
    fourcc = cv.VideoWriter_fourcc(*"XVID")
    out = cv.VideoWriter("output_path.avi", fourcc, 30, (width, height), isColor=True)

    while True:

        if cv.waitKey(1) & 0xFF == ord("q"):
            break
        if cap.isOpened():
            ret, frame = cap.read()
            detection_shape_holder = []
            label_info = []

            if ret:
                tracking_list = predict_func(model, frame)
                for idx, (tracking_details, labels) in enumerate(tracking_list):  # Iterating through all the frame

                    for (track_id, tlwh, t_score), label in zip(tracking_details,
                                                                labels):  # Iterating through each detection in each frame
                        if label not in map_list:
                            continue
                        x1, y1, w, h = tuple(map(int, tlwh))
                        x2, y2 = x1 + w, y1 + h


                        bounding_shapes = [
                            [x1, y1],
                            [x2, y2],
                        ]

                        detection_shape_holder.append(bounding_shapes)
                        label_info.append({
                            "text": f"ID: {label} {int(track_id)} -> {t_score:.2f}",
                            "pos": (x1, y1 - 5),
                            "track_id": track_id,
                        })
                        print(f"{label} {int(track_id)} -> {t_score:.2f}")
                        print(frame_rate)

                    frame = annotate_func(detection_shape_holder, label_info, frame, bounding_color_dict)

                    cv.imshow("video", frame)
                    out.write(frame)

            #detection_to_jason(detection_list)

    cap.release()
    cv.destroyAllWindows()


def annotate_func(bounding_shapes, label_info, frame, bounding_color_dict) -> np.array:
    if len(bounding_shapes) > 0:

        for idx, label in enumerate(label_info):
            track_id = label["track_id"]
            b_, g_, r_ = bounding_color_gen(track_id, bounding_color_dict)
            frame = cv.rectangle(frame, bounding_shapes[idx][0], bounding_shapes[idx][1], (b_, g_, r_), 2, cv.LINE_AA, )
            cv.putText(frame, label["text"], label["pos"], cv.FONT_HERSHEY_SIMPLEX, 0.5, (250, 0, 0), 1)

    return frame


def predict_func(model: YOLO, frames: np.array) -> list:
    tracking_list = list()

    predictions = model.predict(frames, stream=True)

    for prediction in predictions:
        image_shape = prediction.orig_shape
        class_names = prediction.names
        boxes = prediction.boxes
        conf = boxes.conf.cpu().numpy().reshape(-1, 1)
        labels = boxes.cls.cpu().numpy()
        labels = [class_names[label] for label in labels]
        bboxes = prediction.boxes.xyxy.cpu().numpy()

        detection = np.concatenate((bboxes, conf), axis=1)
        online_targets = tracker.update(detection, image_shape, image_shape)  # ByteTracker
        # online_targets = tracker.update(prediction)  # Deepsorttracker
        tracking_details = [(t.track_id, t.tlwh, t.score) for t in online_targets]
        tracking_list.append([tracking_details, labels])
    return tracking_list


def detection_to_jason(detection_list):
    path = os.getcwd()
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    file_path = os.path.join(path, "detections.json")

    with open(file_path, "w") as f:
        json.dump(detection_list, f)
        print("save result succesfully")


def bounding_color_gen(track_id, bounding_color_dict):


    if len(bounding_color_dict) > 0:
        if track_id in bounding_color_dict.keys():
            b_, g_, r_ = bounding_color_dict[track_id][0], bounding_color_dict[track_id][1], \
            bounding_color_dict[track_id][2]
            return b_, g_, r_
        else:
            b_, g_, r_, = random.randint(0, 256), random.randint(0, 256), random.randint(0, 256)
            bounding_color_dict.update({track_id: (b_, g_, r_)})

            print(b_, g_, r_)

            return b_, g_, r_
    return None


class Args:
    track_buffer = 30
    match_thresh = 0.80
    track_thresh = 0.55
    mot20 = True


args = Args()
frame_rate = None
path = "Datasets/ikeja.mp4"
model_path = "Checkpoints/yolo12n.pt"
tracker = None

#tracker = DeepSortTracker(metric_name="euclidean", max_iou_distance=0.8, max_age=30, n_init= 3, max_dist=0.2, nn_budget=100)

if __name__ == "__main__":
    model = YOLO(model_path)
    video_reader(path, model)
