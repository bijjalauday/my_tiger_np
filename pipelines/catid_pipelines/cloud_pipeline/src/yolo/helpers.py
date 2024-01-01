import os

from src.yolo.detect import detect


def inference(
    model, input_path, output_path, enroll=False, augment=False
):  #  video=False,
    """
    This function loads an image/video from the input folder (data/raw/inference_data) and applies the yolo model for inferencing and saves the cropped images
    and corresponding json output file in the output(processed/indference_data) folder.

    Parameters:
        model (str): Path of the yolov5 lite model object
        input_path (str): Path of the input file
        output_path (str): Path of the output file
        top_frames (int): Number of top frames need to send to cloud (frames with higher cofindence score)
        conf_thresh (float): Threshold to reject frames less than the desired confidence value

    Returns:
        None

    """

    # output_folder, filename = create_output_folder(input_path, output_path)
    if enroll:
        detect(
            model,
            input_path,
            img_size=(256, 256),
            project=output_path,  # output_folder,
            augment=augment,
            enroll=enroll,  # if enroll is True, will trigger front side cls model and do enroll augmentations.
            # top_frames=top_frames,
            # name=filename,
            # video=video,
            # conf_thres=conf_thresh,
        )
        if augment:
            print("Frontfacing and sidefacing images for enrollment saved.")
            print("Augmentation for enroll images completed")
        else:
            print("Frontfacing and sidefacing images for enrollment saved.")
