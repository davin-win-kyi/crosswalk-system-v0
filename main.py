from transformers import pipeline
from PIL import Image
import requests
import time

# load pipe
pipe = pipeline(task="depth-estimation", model="LiheYoung/depth-anything-small-hf")

# load image
path = 'C:\Users\davin\PycharmProjects\Depth_Testing\crosswalk_testimage#1.jpg'
image = Image.open(path)

image.save("orig_image.jpg", "JPEG")

start_time = time.time()
# inference
depth = pipe(image)["depth"]

end_time = time.time()

print("Inference time: ", (end_time - start_time))

print("depth: ", depth)

depth.save("depth_image.jpg", "JPEG")


'''''''''
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
'''
