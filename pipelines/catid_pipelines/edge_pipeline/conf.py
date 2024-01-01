import os

base_dir = os.getcwd()
# base_dir = "D:/Cat_id/nano pipeline/yolo_edge_pipeline_exp/yolov5nano/"

# model artifacts path
weights = base_dir + "/artifacts/yolo/weights/best-fp16.tflite"

# path for input data
input_dir = base_dir + "/data/raw/inference_data/"
# path to save output data
output_dir = base_dir + "/data/processed/inference_data/"

image_extensions = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
video_extensions = ["mov", "avi", "mp4", "mpg", "mpeg", "m4v", "wmv", "mkv"]


conf_thresh = 0.6  # Threshold to reject frames less than the desired confidence value
top_frames = 5  # Number of top frames to be saved.
enroll = False  # if videos/images are for pet enrollment.
