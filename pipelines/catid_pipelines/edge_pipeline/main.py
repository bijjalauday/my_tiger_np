import warnings

from conf import *
from helpers import inference

warnings.filterwarnings("ignore")

# Change the directory to current file path
# cur_path = "D:/Cat_id/nano pipeline/yolo_edge_pipeline_exp/yolov5nano/"
# os.chdir(cur_path)


# Call the inference function for yolo model detection on input data
inference(
    weights,
    input_dir,
    output_dir,
    top_frames,
    conf_thresh,
    enroll,
)
