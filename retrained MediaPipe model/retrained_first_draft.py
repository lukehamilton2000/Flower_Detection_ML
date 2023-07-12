#from google.colab import files
import os
import json
#import tensorflow as tf
#assert tf.__version__.startswith('2')

from mediapipe_model_maker import object_detector

train_dataset_path = r"E:\University\Bristol\Dissertation\code\retrained MediaPipe model\tomato_flower_dataset\train"
validation_dataset_path = r"E:\University\Bristol\Dissertation\code\retrained MediaPipe model\tomato_flower_dataset\validation"


train_data = object_detector.Dataset.from_coco_folder(train_dataset_path, cache_dir="/tmp/od_data/train")
validation_data = object_detector.Dataset.from_coco_folder(validation_dataset_path, cache_dir="/tmp/od_data/validation")
print("train_data size: ", train_data.size)
print("validation_data size: ", validation_data.size)
