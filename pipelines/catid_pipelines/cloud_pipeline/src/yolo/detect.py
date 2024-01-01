import json
import os
import random
import time
from pathlib import Path

import albumentations as A
import cv2
import pandas as pd
import torch
import torch.nn.functional as F

from src.yolo.models.common import DetectMultiBackend
from src.yolo.utils.augmentations import classify_transforms
from src.yolo.utils.dataloaders import LoadImages, LoadScreenshots, LoadStreams
from src.yolo.utils.general import (
    Profile,
    check_img_size,
    check_imshow,
    cv2,
    increment_path,
)
from src.yolo.utils.torch_utils import select_device, time_sync


def detect(
    weights="artifacts/yolo/v1.0/model/weights/classification_best.pt",
    source="data/raw/infernce_data/",
    img_size=(256, 256),
    # data=ROOT / 'data/coco128.yaml',
    device="cpu",
    view_img=False,
    save_txt=True,
    nosave=True,
    classes=[0, 1],
    augment=True,
    enroll=False,
    update=True,
    project="data/processed/infernce_data",
    exist_ok=True,
):
    weights, view_img, save_txt, imgsz = (
        weights,
        view_img,
        save_txt,
        img_size,
    )
    # save_img = not nosave and not source.endswith(".txt")  # save inference images

    webcam = (
        source.isnumeric()
        or source.endswith(".txt")
        or source.lower().startswith(("rtsp://", "rtmp://", "http://", "https://"))
    )

    device = select_device(device)
    half = device.type != "cpu"  # half precision only supported on CUDA

    # Load model
    device = select_device(device)
    model = DetectMultiBackend(weights, device=device, fp16=half)
    stride, names, pt = model.stride, model.names, model.pt
    imgsz = check_img_size(imgsz, s=stride)  # check image size

    # Dataloader
    bs = 1  # batch_size

    if enroll:
        project_ = "data/processed/enroll_data"
        source_ = "data/raw/enroll_data"

        for houses in os.listdir(source_):
            for pettypes in os.listdir(f"{source_}/{houses}"):
                for pets in os.listdir(f"{source_}/{houses}/{pettypes}"):
                    for top_frames in os.listdir(
                        f"{source_}/{houses}/{pettypes}/{pets}"
                    ):
                        source = f"{source_}/{houses}/{pettypes}/{pets}/{top_frames}"
                        project = f"{project_}/{houses}/{pettypes}/{pets}/"
                        if webcam:
                            view_img = check_imshow(warn=True)
                            dataset = LoadStreams(
                                source,
                                img_size=imgsz,
                                stride=stride,
                                transforms=classify_transforms(imgsz[0]),
                            )
                            bs = len(dataset)
                        # elif screenshot:
                        #     dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
                        else:
                            dataset = LoadImages(
                                source,
                                img_size=imgsz,
                                transforms=classify_transforms(imgsz[0]),
                                stride=stride,
                            )
                        vid_path, vid_writer = [None] * bs, [None] * bs

                        # Run inference
                        model.warmup(
                            imgsz=(1 if pt or model.triton else bs, 3, imgsz)
                        )  # warmup
                        seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
                        t0 = time.time()
                        # output = []
                        pred_dect = dict()
                        # d_images = dict()
                        for path, im, im0s, vid_cap, s in dataset:
                            with dt[0]:
                                im = torch.Tensor(im).to(model.device)
                                im = (
                                    im.half() if model.fp16 else im.float()
                                )  # uint8 to fp16/32
                                if len(im.shape) == 3:
                                    im = im[None]  # expand for batch dim

                            # Inference
                            with dt[1]:
                                t1 = time_sync()

                                results = model(im)

                            # NMS
                            with dt[2]:
                                pred = F.softmax(results, dim=1)  # probabilities

                                t2 = time_sync()

                            if path.split(".")[-1] in [
                                "mov",
                                "avi",
                                "mp4",
                                "mpg",
                                "mpeg",
                                "m4v",
                                "wmv",
                                "mkv",
                            ]:
                                video = True
                            else:
                                video = False

                            # Process predictions
                            for i, prob in enumerate(pred):  # per image
                                seen += 1
                                if webcam:  # batch_size >= 1
                                    p, im0, frame = (
                                        path[i],
                                        im0s[i].copy(),
                                        dataset.count,
                                    )
                                    s += f"{i}: "

                                # elif video:
                                #     p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                                #     p = Path(p)
                                #     filename = p.stem + "_" + str(frame)
                                #     save_dir = increment_path(Path(project) / p.stem, exist_ok=exist_ok)
                                #     (save_dir).mkdir(parents=True, exist_ok=True)

                                else:
                                    p, im0, frame = (
                                        path,
                                        im0s.copy(),
                                        getattr(dataset, "frame", 0),
                                    )
                                    p = Path(p)
                                    # save_dir = increment_path(Path(project), exist_ok=exist_ok)
                                    # (save_dir).mkdir(parents=True, exist_ok=True)

                                s += "%gx%g " % im.shape[2:]  # print string
                                p = Path(p)  # to Path

                                top5i = prob.argsort(0, descending=True)[
                                    :5
                                ].tolist()  # top 5 indices
                                s += f"{', '.join(f'{names[j]} {prob[j]:.2f}' for j in top5i)}, "
                                # textt = {names[j]: prob[j].numpy()[0] for j in top5i}

                                res_dic = dict()
                                for j in top5i:
                                    res_dic[names[j]] = float(prob[j].numpy())
                                    save_dir = increment_path(
                                        Path(project) / names[j], exist_ok=exist_ok
                                    )
                                    (save_dir).mkdir(parents=True, exist_ok=True)

                                # d[p.stem] = res_dic
                                filename = list(res_dic.keys())[0]
                                if filename == "front_facing":
                                    cv2.imwrite(
                                        str(project)
                                        + "/front_facing/"
                                        + p.stem
                                        + ".jpg",
                                        im0,
                                    )
                                else:
                                    cv2.imwrite(
                                        str(project)
                                        + "/side_facing/"
                                        + p.stem
                                        + ".jpg",
                                        im0,
                                    )

                                pred_dect[p.stem] = res_dic

                                if (
                                    augment
                                ):  # if True , do augmentations for enrolling (rotation and brightness)
                                    aug_save_dir = increment_path(
                                        Path(project) / "enroll_with_aug/",
                                        exist_ok=exist_ok,
                                    )
                                    (aug_save_dir).mkdir(parents=True, exist_ok=True)

                                    transform = A.Compose(
                                        [  # 40 degree rotation
                                            A.Rotate(p=1, limit=[40, 40]),
                                        ]
                                    )

                                    transform1 = A.Compose(
                                        [  # 320 degree rotation
                                            A.Rotate(p=1, limit=[320, 320]),
                                        ]
                                    )

                                    brightness_factor = 1.5
                                    reduced_factor = 0.75
                                    brightened_image = cv2.convertScaleAbs(
                                        im0, alpha=brightness_factor, beta=0
                                    )
                                    reduced_image = cv2.convertScaleAbs(
                                        im0, alpha=reduced_factor, beta=0
                                    )
                                    transformed = transform(image=im0)
                                    transformed_image = transformed["image"]
                                    transformed1 = transform1(image=im0)
                                    transformed_image1 = transformed1["image"]

                                    cv2.imwrite(f"{aug_save_dir}/{p.stem}.jpg", im0)
                                    cv2.imwrite(
                                        f"{aug_save_dir}/{p.stem}_40_rotated.jpg",
                                        transformed_image,
                                    )
                                    cv2.imwrite(
                                        f"{aug_save_dir}/{p.stem}_320_rotated.jpg",
                                        transformed_image1,
                                    )

                                    cv2.imwrite(
                                        f"{aug_save_dir}/{p.stem}_brightness.jpg",
                                        brightened_image,
                                    )
                                    cv2.imwrite(
                                        f"{aug_save_dir}/{p.stem}_brightness_reduced.jpg",
                                        reduced_image,
                                    )
                                    # print("Completed augmentations for enrolling.")
                                else:
                                    enrol_save_dir = increment_path(
                                        Path(project) / "enroll_images/",
                                        exist_ok=exist_ok,
                                    )
                                    (enrol_save_dir).mkdir(parents=True, exist_ok=True)

                                    cv2.imwrite(f"{enrol_save_dir}/{p.stem}.jpg", im0)

                                print(f"{s}Done. ({t2 - t1:.3f}s)")

                            #         # Stream results
                            #         # im0 = annotator.result()

                            if save_txt:  # Write to file
                                # file_path = (project + "/front_side_predictions.json" if enroll else project + "/predictions.json")
                                file_path = project + "/front_side_predictions.json"
                                with open(file_path, "w") as json_file:
                                    json.dump(pred_dect, json_file)

                        print(f"Done. ({time.time() - t0:.3f}s)")
