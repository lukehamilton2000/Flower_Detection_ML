#from google.colab import files
import os
import json
#import tensorflow as tf
#assert tf.__version__.startswith('2')

#from mediapipe_model_maker import object_detector

train_dataset_path = r"E:\University\Bristol\Dissertation\code\retrained MediaPipe model\tomato_flower_dataset\train"
validation_dataset_path = r"E:\University\Bristol\Dissertation\code\retrained MediaPipe model\tomato_flower_dataset\validation"



with open(os.path.join(train_dataset_path, r"E:\University\Bristol\Dissertation\code\retrained MediaPipe model\tomato_flower_dataset\labels.json"), "r") as f:
  labels_json = json.load(f)
for category_item in labels_json["categories"]:
  print(f"{category_item['id']}: {category_item['name']}")
