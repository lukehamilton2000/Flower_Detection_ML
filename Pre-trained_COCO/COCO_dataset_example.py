import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
from visualizer import visualize




# image on pc of cat and dog
# IMAGE_FILE = r'E:\University\Bristol\Dissertation\code\object detection\cat_and_dog.jpg'

# image of tomato flower
# IMAGE_FILE = r'E:\University\Bristol\Dissertation\code\object detection\tomato_flower.png'

# image of another dog
# IMAGE_FILE = r'E:\University\Bristol\Dissertation\code\object detection\dog2.jpg'

# image of human
# IMAGE_FILE = r'E:\University\Bristol\Dissertation\code\object detection\human.jpg'

# image of purple flower
IMAGE_FILE = r'E:\University\Bristol\Dissertation\code\object detection\purple_flower.jpg'

img = cv2.imread(IMAGE_FILE)
cv2.imshow('img', img)

# STEP 2: Create an ObjectDetector object.
base_options = python.BaseOptions(model_asset_path=r'E:\University\Bristol\Dissertation\code\object detection\efficientdet_lite0.tflite')
options = vision.ObjectDetectorOptions(base_options=base_options,
                                       score_threshold=0.5)
detector = vision.ObjectDetector.create_from_options(options)

# STEP 3: Load the input image.
image = mp.Image.create_from_file(IMAGE_FILE)

# STEP 4: Detect objects in the input image.
detection_result = detector.detect(image)

# STEP 5: Process the detection result. In this case, visualize it.
image_copy = np.copy(image.numpy_view())
annotated_image = visualize(image_copy, detection_result)
rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
cv2.imshow('finished', rgb_annotated_image)
cv2.waitKey(10000)
