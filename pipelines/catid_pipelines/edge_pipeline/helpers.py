import os
import time

import cv2

from detect import *


def inference(
    model,
    input_path,
    output_path,
    top_frames,
    conf_thresh,
    # target_fps,
    enroll=False,
    # fps_reduction=False,
):
    """
    This function loads an image/video from the input folder (data/raw/inference_data) and applies the yolo model for inferencing and saves the cropped images
    and corresponding json output file in the output(processed/indference_data) folder.

    Parameters:
        model (str): Path of the yolov5 lite model object
        input_path (str): Path of the input file
        output_path (str): Path of the output file
        top_frames (int): Number of top frames need to send to cloud (frames with higher cofindence score)
        conf_thresh (float): Threshold to reject frames less than the desired confidence value
        enroll (bool): True if the session belongs to enrollment


    Returns:
        None

    """

    if enroll:
        print("Preparing for enrolling the pets...")
    else:
        print("Preparing for inference...")

    detect(
        model,
        input_path,
        img_size=256,
        project=output_path,  # output_folder,
        top_frames=top_frames,
        conf_thres=conf_thresh,
        enroll=enroll,
    )
