import os
import cv2


def video_to_images(video_path, image_output_path, images_per_sec=1, image_prefix=""):
    """Extract images from a video and write them to a specified directory.

    Parameters
    ----------
    video_path : str
        path to video file
    image_output_path : str
        output image base path
    images_per_sec : int
        images to be extracted from each second of video
    image_prefix : str
        common prefix for images
    """
    vidcap = cv2.VideoCapture(video_path)
    vid_fps = round(vidcap.get(cv2.CAP_PROP_FPS))
    images_per_sec = vid_fps if images_per_sec > vid_fps else images_per_sec
    count = 0
    while vidcap.isOpened():
        success, image = vidcap.read()
        if success:
            if count % (vid_fps / images_per_sec) == 0:
                cv2.imwrite(os.path.join(image_output_path, f'{image_prefix}_video_frame_{count}.JPG'), image)
            count += 1
        else:
            break
    cv2.destroyAllWindows()
    vidcap.release()
