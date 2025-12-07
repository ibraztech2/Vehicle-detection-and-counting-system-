import os
import json
import cv2 as cv
import numpy as np
from deepsort import DeepSortTracker
from ultralytics import YOLO
import time

from bytetrack import yolox
from bytetrack.yolox.tracker.byte_tracker import BYTETracker


def video_reader(path, model):
    global tracker
    start_time = time.time()
    cap = cv.VideoCapture(path)

    height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    frame_rate = int(cap.get(cv.CAP_PROP_FPS))
    width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))

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
                for idx, (tracking_details, labels) in enumerate(tracking_list):
                    for (track_id, tlwh, t_score), label in zip(tracking_details, labels):
                        x1, y1, w, h = tuple(map(int, tlwh))
                        x2, y2 = x1 + w, y1 + h

                        bounding_shapes = [
                            [x1, y1],
                            [x2, y1],
                            [x2, y2],
                            [x1, y2],
                        ]

                        detection_shape_holder.append(bounding_shapes)
                        label_info.append({
                            "text": f"ID: {label} {int(track_id)} -> {t_score:.2f}",
                            "pos": (x1, y1 - 5),
                        })

                    frame = annotate_func(detection_shape_holder, label_info, frame)

                    cv.imshow("video", frame)
                    out.write(frame)


            #detection_to_jason(detection_list)

    cap.release()
    cv.destroyAllWindows()


def annotate_func(bounding_shapes, label_info, frame) -> np.array:
    if len(bounding_shapes) > 0:

        frame = cv.polylines(frame, [np.array(p,np.int32) for p in bounding_shapes], True, (0, 200, 0), 2)

        for label in label_info:
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


def time_counter(start_time, duration, ):
    end_time = time.time()
    if duration == end_time - start_time:
        return True


class Args:
    track_buffer = 30
    match_thresh = 0.85
    track_thresh = 0.55
    mot20 = False


args = Args()
frame_rate = None
path = "test_video.mp4"
tracker  = None

#tracker = DeepSortTracker(metric_name="euclidean", max_iou_distance=0.8, max_age=30, n_init= 3, max_dist=0.2, nn_budget=100)

if __name__ == "__main__":
    model = YOLO("yolov8n.pt")
    video_reader(path, model)
