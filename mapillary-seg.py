from ultralytics import YOLO
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import requests
import time
import numpy as np

import cv2

# this is the checkpoint that we will be using for object detection
file = "mapillary-seg-v0.pt"

# model = YOLO("yolov8n.yaml")
# model.load_state_dict(torch.load(file))

# this is for OBJECT DETECTION
model = YOLO(file)

# this is for the DEPTH model
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

img_path = ('C:\\Users\\davin\\PycharmProjects\\Depth_Testing\\crosswalk#6.jpg')
results = model(img_path)

image = Image.open(img_path)

print("Prediction: ", results)

# GET ALL PREDICTION DATA FOR OBJECT DETECTION
boxes = results[0].boxes.xyxy.tolist()
classes = results[0].boxes.cls.tolist()
names = results[0].names

for name in names:
    print(names[name])


draw = ImageDraw.Draw(image)
# this is for the formatting of the text
font_path = "arialbd.ttf"  # 'arialbd.ttf' is typically Arial Bold. Adjust the path to where the font is located on your system.
font_size = 15
font = ImageFont.truetype(font_path, font_size)


for detection in range(len(boxes)):
    bbox = boxes[detection]

    object_name = names[classes[detection]]

    print("Class name: ", object_name)

    print("Bounding box: ", bbox)

    # here we can check over to see the average depth for the given
    # object in the image, and will add the object into a list which
    # we will check over
    # round the values by an int()

    x1 = int(bbox[0])
    y1 = int(bbox[1])
    x2 = int(bbox[2])
    y2 = int(bbox[3])

    # let's determine the threshold first, then
    # we can discuss how to determine direction (will that also have a threshold?)
    # what are some edge cases to consider
    # etc. concerns

    crosswalk_types = ["construction--flat--crosswalk-plain",
                       "marking--discrete--crosswalk-zebra",
                       "marking-only--discrete--crosswalk-zebra"]

    pedestrian_traffic_signal_types = ["object--traffic-light--pedestrians",
                                       "object--support--traffic-sign-frame",
                                       "object--traffic-light--general-single",
                                       "object--traffic-light--general-upright",
                                       "object--traffic-light--general-horizontal",
                                       "object--traffic-light--other",
                                       ]
    if object_name == "construction--flat--road":
        # draw the bounding box of the image as well
        # Draw the bounding box
        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline="red", width=2)
        # Draw the label on the image
        label = f"{object_name}"  # Format the label with the object name and mean depth
        draw.text((x1, y1), label, fill="yellow", font=font)  # You can change the text color and position as needed

image.save("annotated_image.jpg", "JPEG")  # Save the annotated image