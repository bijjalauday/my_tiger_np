import os
import argparse
from ta_pet_id.core.utils import get_package_path
from ta_pet_id.core.base_utils import silence_all_warnings
import process

config_path = os.path.join(get_package_path(), "../production/conf/config.yml")

parser = argparse.ArgumentParser(description="Pet Recognition Processes")
parser.add_argument(
    "process",
    type=str,
    help="run the pet recognition module processes - enroll, inference or train",
    choices=["enroll", "inference", "train-effnet", "train-yolo"],
)

if __name__ == "__main__":
    silence_all_warnings()
    arg, sub_args = parser.parse_known_args()
    if arg.process == "enroll":
        parser = argparse.ArgumentParser()
        parser.add_argument('re_enroll', type=int, nargs='?', const=0)
        arg = parser.parse_args(sub_args)
        print(__file__, '[+] Enrollment: STARTED')
        print("re_enroll is set to:", bool(arg.re_enroll))
        process.run_enrollment(config_path, re_enroll=bool(arg.re_enroll))
        print(__file__, '[+] Enrollment: COMPLETED')
    elif arg.process == "inference":
        print(__file__, '[+] Inference: STARTED')
        process.run_inference(config_path)
        print(__file__, '[+] Inference: COMPLETED')
    elif arg.process == "train-effnet":
        print(__file__, '[+] EfficientNetB2 Training: STARTED')
        process.train_efficientnet_model(config_path)
        print(__file__, '[+] EfficientNetB2 Training: COMPLETED')
    elif arg.process == "train-yolo":
        print(__file__, '[+] YOLOv5 Training: STARTED')
        process.train_yolo_model(config_path)
        print(__file__, '[+] YOLOv5 Training: COMPLETED')
    else:
        print("Invalid choice: choose one from ['enroll', 'inference', 'train-effnet', 'train-yolo]")
