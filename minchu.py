from ultralytics import YOLO
from transformers import pipeline
from PIL import Image, ImageDraw, ImageFont
import requests
import time
import numpy as np

import cv2


# this is to get the range of hue values for a color
def get_limits(color):

    c = np.uint8([[color]])
    hsvC = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)

    lowerLimit = hsvC[0][0][0] - 10, 100, 100
    upperLimit = hsvC[0][0][0] + 10, 255, 255

    lowerLimit = np.array(lowerLimit, dtype=np.uint8)
    upperLimit = np.array(upperLimit, dtype=np.uint8)

    return lowerLimit, upperLimit



# this is the checkpoint that we will be using for object detection
file = "best.pt"

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

# GET ALL DATA FROM DEPTH MODEL
depth = pipe(image)["depth"]
depth.save("depth_image.jpg", "JPEG")
depth_image = np.array(depth)

# print("CLASSES: ", classes)
# print("NAMES: ", names)
# Create an ImageDraw object to annotate the image
# You can use ImageDraw.Draw on an image to have an object
# that you can use to draw onto the object in which
# you want to deal with
draw = ImageDraw.Draw(image)

# this is for the formatting of the text
font_path = "arialbd.ttf"  # 'arialbd.ttf' is typically Arial Bold. Adjust the path to where the font is located on your system.
font_size = 15
font = ImageFont.truetype(font_path, font_size)

# these are all of the crosswalks we detected, we will care about the one
# that is closest to the user
crosswalks = []

# these are the pedestrian traffic signals
pedestrian_traffic_signals = []

# these are the objects we should notify about
notify_objects = []

# now we can loop over each of the objects in the image and we can
# check to see how close the objects are, we can set a threshold as
# needed
# Example parsing (details will depend on the actual output format of your model)
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
    if pedestrian_traffic_signal_types.__contains__(object_name) or crosswalk_types.__contains__(object_name):
        # draw the bounding box of the image as well
        # Draw the bounding box
        draw.rectangle((bbox[0], bbox[1], bbox[2], bbox[3]), outline="red", width=2)
        # Draw the label on the image
        label = f"{object_name}"  # Format the label with the object name and mean depth
        draw.text((x1, y1), label, fill="yellow", font=font)  # You can change the text color and position as needed

    # get the depth of each of these objects
    depth_matrix = depth_image[x1:x2, y1:y2]

    mean_depth = np.mean(depth_matrix)

    # deal with crosswalks
    if crosswalk_types.__contains__(object_name):
        crosswalks.append((bbox, mean_depth))

    if pedestrian_traffic_signal_types.__contains__(object_name):
        pedestrian_traffic_signals.append((bbox, mean_depth))

    # here we will be determing whether we want to notify a user about an object or not
    if not crosswalk_types.__contains__(object_name) and not pedestrian_traffic_signal_types.__contains__(object_name):

        depth_threshold = 75

        if mean_depth >= depth_threshold:

            x_coordinate_of_object = (x1 + x2) // 2

            width, height = image.size

            middle_x = width // 2

            # dummy thresholds, we will need to do some testing
            # to solidify some threshold values that make a lot more sense
            # I think that a double threshold will make the most amount of sense
            # one where it is very close
            # one for where it is close by but as urgent as something that is very close
            middle_threshold = 0.10
            close_threshold = 0.25

            x_diff = middle_x - x_coordinate_of_object

            percent_from_middle = (x_diff * 1.0) / width

            direction = ""
            if x_diff < 0 and percent_from_middle > 0.1:
                if percent_from_middle < 0.25:
                    direction = "close-right"
                else:
                    direction = "right"
            elif x_diff > 0 and percent_from_middle > 0.1:
                if percent_from_middle < 0.25:
                    direction = "close-left"
                else:
                    direction = "left"
            else:
                direction = "middle"


            close_string = ""

            if mean_depth >= 90:
                close_string = "very close"
            else:
                close_string = "close by"


            object = (object_name, direction, close_string)

            notify_objects.append(object)

# these are all of the objects that we will notify the user about
# we will alert the user with a specific message about the object
# that passed the threshold values
for object in notify_objects:
    object_name = object[0]
    direction = object[1]
    close_string = object[2]

    object_message = "There is a " + object_name + " that is to the " + direction + " that is " + close_string

    print(object_message)


# here we will deal with the crosswalk data, and will base it off of the
# middle coordinate of the crosswalk
max_depth_crosswalk = 0
crosswalk_found = False
crosswalk_max = None
for crosswalk in crosswalks:
    crosswalk_found = True
    bbox = crosswalk[0]
    mean_depth = crosswalk[1]
    if mean_depth > max_depth_crosswalk:
        max_depth_crosswalk = mean_depth
        crosswalk_max = bbox

crosswalk_notification = ""
# now with the bounding box, we will use it to determine if the user is
# straying to the left of the right of the crosswalk
if crosswalk_max is not None:
    bbox = crosswalk_max
    x1 = int(bbox[0])
    x2 = int(bbox[1])
    average_x = (x1 + x2) / 2
    # now we can try to determine if the user
    # is away from the center of the image or not
    width, height = image.size
    center = width // 2

    # Calculate the distance from the center of the crosswalk to the center of the image
    distance_from_center = abs(average_x - center)

    width_of_sidewalk = (x2 -x1)

    # Calculate 25% of the image's width
    threshold = width_of_sidewalk * 0.4

    # Check if the distance from the center is at least 25% of the image's width
    if distance_from_center >= threshold and average_x < center:
        crosswalk_notification = "You are a bit far to the left of the crosswalk"
    elif distance_from_center >= threshold and average_x > center:
        crosswalk_notification = "You are a bit far to the right of the crosswalk"


# we want to work with the HSV color space for color detection
'''''''''''
# Hue, Saturation, Value
'''

# here we will deal with the pedestrian traffic signal information and will
# check to see what the traffic signal has if there are any
# if there is one, we will run a color model and possibly LLaVA or some sort
# of text detection model
pts_found = False
max_depth = 0
pts_max = None
for pts in pedestrian_traffic_signals:
    pts_found = True
    # get information about the bounding box and the depth
    bbox = pts[0]
    mean_depth = pts[1]
    # now we can set a new max pedestrian signal if needed
    if mean_depth > max_depth:
        max_depth = mean_depth
        pts_max = bbox

# now we can get information from this pedestrian traffic signal
pts_message = "No traffic signal information available"

if pts_max is not None:
    # here we will be doing color detection of the
    # pedestrian traffic signal
    print("")
    cropBox = (int(pts_max[0]), int(pts_max[1]), int(pts_max[2]), int(pts_max[3]))
    cropped_image = image.crop(cropBox)
    # Convert the PIL Image to a NumPy array
    cropped_image_np = np.array(cropped_image)
    hsvImage = cv2.cvtColor(cropped_image_np, cv2.COLOR_BGR2HSV)

    # two values for the bound of an image
    red = [255, 0, 0]
    white = [255, 255, 255]

    # figure out what percent of the image is red
    lower_limit_red, upper_limit_red = get_limits(color=red)
    mask = cv2.inRange(hsvImage, lower_limit_red, upper_limit_red)
    red_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    percentage_red = (red_pixels / total_pixels) * 100

    # figure out what percent of the image is white
    lower_limit_white, upper_limit_white = get_limits(color=white)
    mask = cv2.inRange(hsvImage, lower_limit_white, upper_limit_white)
    white_pixels = cv2.countNonZero(mask)
    total_pixels = mask.size
    percentage_white = (red_pixels / total_pixels) * 100

    if percentage_white >= 10:
        pts_message = "You can cross the street"

    # here we will need to get some text information
    else:
        pts_message = "It is advisable to not cross the street"

        # IMPORTANT!!!
        # here let's try to include the code nessecary to use the Google Vision Cloud API for OCR




'''''''''
at the very end, we need to give information about:
- pedestrian traffic signal
- crosswalk
- objects that are close by 
'''
print(crosswalk_notification)
print(pts_message)



image.save("annotated_image.jpg", "JPEG")  # Save the annotated image