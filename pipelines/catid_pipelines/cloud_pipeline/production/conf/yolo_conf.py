import os

base_dir = os.getcwd()
# base_dir = "D:/Cat_id/cloud_pipeline/"
# D:\Cat_id\cloud_pipeline\src\yolo

# model artifacts path
weights = base_dir + "/artifacts/yolo/v1.0/model/weights/classification_best.pt"

# path for input data
input_dir = base_dir + "/data/raw/infernce_data/"
# path to save output data
output_dir = base_dir + "/data/processed/"

image_extensions = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
video_extensions = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]

enroll = False
augment = True
