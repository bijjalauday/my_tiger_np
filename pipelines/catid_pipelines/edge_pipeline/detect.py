import json
import os
import time
from pathlib import Path

import cv2
import pandas as pd
import torch

from models.common import DetectMultiBackend
from utils.dataloaders import LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    Profile,
    check_img_size,
    check_imshow,
    cv2,
    increment_path,
    non_max_suppression,
    scale_boxes,
    xyxy2xywh,
)
from utils.torch_utils import select_device, time_sync


def join_fn(col):
    col = col.split("_")[:-1]
    return "_".join(col)


def detect(
    weights="artifacts/yolo/weights/best-fp16.tflite",
    source="/data/raw/inference_data/",
    img_size=256,
    conf_thres=0.75,
    iou_thres=0.4,
    device="cpu",
    view_img=False,
    save_txt=True,
    save_conf=True,
    nosave=True,
    classes=[0, 1],
    agnostic_nms=True,
    augment=False,
    update=True,
    project="/data/processed/inference_data",
    top_frames=5,
    exist_ok=True,
    enroll=False,
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

    # Run inference
    # Dataloader
    bs = 1  # batch_size
    model.warmup(imgsz=(1 if pt or model.triton else bs, 3, imgsz))  # warmup
    seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
    t0 = time.time()

    if enroll:
        project_ = "data/processed/enroll_data"
        source_ = "data/raw/enroll_data"

        for houses in os.listdir(source_):
            for pettypes in os.listdir(f"{source_}/{houses}"):
                for pets in os.listdir(f"{source_}/{houses}/{pettypes}"):
                    source = f"{source_}/{houses}/{pettypes}/{pets}"
                    project = f"{project_}/{houses}/{pettypes}/{pets}"

                    if webcam:
                        view_img = check_imshow(warn=True)
                        dataset = LoadStreams(
                            source, img_size=imgsz, stride=stride, auto=pt
                        )
                        bs = len(dataset)
                    # elif screenshot:
                    #     dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
                    else:
                        dataset = LoadImages(
                            source, img_size=imgsz, stride=stride, auto=pt
                        )
                    vid_path, vid_writer = [None] * bs, [None] * bs

                    output = []
                    d = dict()
                    d_images = dict()
                    t5 = time.time()
                    for path, im, im0s, vid_cap, s in dataset:
                        with dt[0]:
                            im = torch.from_numpy(im).to(model.device)
                            im = (
                                im.half() if model.fp16 else im.float()
                            )  # uint8 to fp16/32
                            im /= 255  # 0 - 255 to 0.0 - 1.0
                            if len(im.shape) == 3:
                                im = im[None]  # expand for batch dim

                        # Inference
                        with dt[1]:
                            t1 = time_sync()

                            pred = model(im, augment=augment)

                        # NMS
                        with dt[2]:
                            pred = non_max_suppression(
                                pred, conf_thres, iou_thres, classes, agnostic_nms
                            )

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
                        # print("pred :", pred)

                        for i, det in enumerate(pred):  # per image
                            seen += 1
                            if webcam:  # batch_size >= 1
                                p, im0, frame = path[i], im0s[i].copy(), dataset.count
                                s += f"{i}: "

                            elif video:
                                p, im0, frame = (
                                    path,
                                    im0s.copy(),
                                    getattr(dataset, "frame", 0),
                                )
                                p = Path(p)
                                if len(det):
                                    filename = p.stem + "_" + str(frame)
                                    save_dir = increment_path(
                                        Path(project) / p.stem, exist_ok=exist_ok
                                    )
                                    (save_dir).mkdir(parents=True, exist_ok=True)

                            else:
                                p, im0, frame = (
                                    path,
                                    im0s.copy(),
                                    getattr(dataset, "frame", 0),
                                )
                                p = Path(p)
                                if len(det):
                                    save_dir = increment_path(
                                        Path(project) / p.stem, exist_ok=exist_ok
                                    )
                                    (save_dir).mkdir(parents=True, exist_ok=True)

                            s += "%gx%g " % im.shape[2:]  # print string
                            gn = torch.tensor(im0.shape)[
                                [1, 0, 1, 0]
                            ]  # normalization gain whwh
                            p = Path(p)  # to Path

                            if len(det):
                                di = {}
                                # Rescale boxes from img_size to im0 size
                                det[:, :4] = scale_boxes(
                                    im.shape[2:], det[:, :4], im0.shape
                                ).round()

                                # Print results
                                for c in det[:, 5].unique():
                                    n = (det[:, 5] == c).sum()  # detections per class
                                    s += f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string

                                # Write results
                                count = 0
                                print("count :", count)
                                for *xyxy, conf, cls in reversed(det):
                                    if save_txt:  # Write to file
                                        xywh = (
                                            (
                                                xyxy2xywh(torch.tensor(xyxy).view(1, 4))
                                                / gn
                                            )
                                            .view(-1)
                                            .tolist()
                                        )  # normalized xywh
                                        line = (
                                            (cls, *xywh, conf)
                                            if save_conf
                                            else (cls, *xywh)
                                        )  # label format

                                        xmin = int(xyxy[0].item())
                                        ymin = int(xyxy[1].item())
                                        xmax = int(xyxy[2].item())
                                        ymax = int(xyxy[3].item())
                                        if video:
                                            filename = p.stem + "_" + str(frame)
                                        else:
                                            filename = p.stem + "_" + str(count)
                                        new_img = im0[ymin:ymax, xmin:xmax]
                                        cv2.imwrite(
                                            str(save_dir) + "/" + filename + ".jpg",
                                            new_img,
                                        )
                                        d_images[filename] = new_img
                                        category = int(cls.item())

                                        di[f"{filename}"] = dict()
                                        di[f"{filename}"]["bbox_coordinates"] = [
                                            xmin,
                                            ymin,
                                            xmax,
                                            ymax,
                                        ]
                                        di[f"{filename}"][
                                            "confidence_score"
                                        ] = conf.item()

                                        if category == 0:
                                            di[f"{filename}"]["category"] = "cat"
                                        else:
                                            di[f"{filename}"]["category"] = "dog"

                                    count += 1
                                d[
                                    (
                                        sorted(
                                            di.items(),
                                            key=lambda x: x[1]["confidence_score"],
                                            reverse=True,
                                        )[0]
                                    )[0]
                                ] = (
                                    sorted(
                                        di.items(),
                                        key=lambda x: x[1]["confidence_score"],
                                        reverse=True,
                                    )[0]
                                )[
                                    1
                                ]

                            print(f"{s}Done. ({t2 - t1:.3f}s)")

                            # Stream results
                            # im0 = annotator.result()
                        if len(d):
                            if save_txt:  # Write to file
                                file_path = project + "/predictions.json"
                                with open(file_path, "w") as json_file:
                                    json.dump(d, json_file)
                    t4 = time.time()
                    print(f"completed (Detection and saved). ({t4 - t5:.3f}s)")
                    if len(d):
                        pred_df = (
                            pd.DataFrame(d)
                            .transpose()
                            .reset_index()
                            .rename(columns={"index": "image_name"})
                        )
                        pred_df["name"] = pred_df["image_name"].apply(join_fn)
                        pred_df = pred_df[pred_df["category"] == pettypes].sort_values(
                            by=["name", "confidence_score"], ascending=[True, False]
                        )
                        top_cat_scores = pred_df.groupby("name").head(top_frames)
                        top_cat_scores.reset_index(drop=True, inplace=True)
                        top_cat_scores = top_cat_scores.drop("name", axis=1)

                        save_dir = increment_path(
                            Path(project) / f"top_frames", exist_ok=exist_ok
                        )  # increment run
                        (save_dir).mkdir(parents=True, exist_ok=True)

                        for i in top_cat_scores["image_name"].values:
                            cv2.imwrite(str(save_dir) + "/" + i + ".jpg", d_images[i])

                        top_cat_scores.to_excel(str(project) + f"/top_predictions.xlsx")
                        print(f"Top frames saved. ({time.time() - t4:.3f}s)")
                    print(f"Done. ({time.time() - t0:.3f}s)")

    else:
        for houses in os.listdir(source):
            # file_names = []
            source_ = f"{source}/{houses}/"
            project_ = f"{project}/{houses}/"
            # file_names.extend(
            #     [".".join(i.split(".")[:-1]) for i in os.listdir(source_)]
            # )
            # break
            if webcam:
                view_img = check_imshow(warn=True)
                dataset = LoadStreams(source_, img_size=imgsz, stride=stride, auto=pt)
                bs = len(dataset)
            # elif screenshot:
            #     dataset = LoadScreenshots(source, img_size=imgsz, stride=stride, auto=pt)
            else:
                dataset = LoadImages(source_, img_size=imgsz, stride=stride, auto=pt)
            vid_path, vid_writer = [None] * bs, [None] * bs
            output = []
            d = dict()
            d_images = dict()
            t5 = time.time()
            for path, im, im0s, vid_cap, s in dataset:
                with dt[0]:
                    im = torch.from_numpy(im).to(model.device)
                    im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
                    im /= 255  # 0 - 255 to 0.0 - 1.0
                    if len(im.shape) == 3:
                        im = im[None]  # expand for batch dim

                # Inference
                with dt[1]:
                    t1 = time_sync()

                    pred = model(im, augment=augment)

                # NMS
                with dt[2]:
                    pred = non_max_suppression(
                        pred, conf_thres, iou_thres, classes, agnostic_nms
                    )

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

                for i, det in enumerate(pred):  # per image
                    seen += 1
                    if webcam:  # batch_size >= 1
                        p, im0, frame = path[i], im0s[i].copy(), dataset.count
                        s += f"{i}: "

                    elif video:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                        p = Path(p)
                        filename = p.stem + "_" + str(frame)
                        if len(det):
                            save_dir = increment_path(
                                Path(project_) / p.stem, exist_ok=exist_ok
                            )
                            (save_dir).mkdir(parents=True, exist_ok=True)

                    else:
                        p, im0, frame = path, im0s.copy(), getattr(dataset, "frame", 0)
                        p = Path(p)
                        if len(det):
                            save_dir = increment_path(
                                Path(project_) / p.stem, exist_ok=exist_ok
                            )
                            (save_dir).mkdir(parents=True, exist_ok=True)

                    s += "%gx%g " % im.shape[2:]  # print string
                    gn = torch.tensor(im0.shape)[
                        [1, 0, 1, 0]
                    ]  # normalization gain whwh
                    p = Path(p)  # to Path

                    if len(det):
                        di = {}
                        # Rescale boxes from img_size to im0 size
                        det[:, :4] = scale_boxes(
                            im.shape[2:], det[:, :4], im0.shape
                        ).round()

                        # Print results
                        for c in det[:, 5].unique():
                            n = (det[:, 5] == c).sum()  # detections per class
                            s += (
                                f"{n} {names[int(c)]}{'s' * (n > 1)}, "  # add to string
                            )

                        # Write results
                        count = 0
                        for *xyxy, conf, cls in reversed(det):
                            if save_txt:  # Write to file
                                xywh = (
                                    (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn)
                                    .view(-1)
                                    .tolist()
                                )  # normalized xywh
                                line = (
                                    (cls, *xywh, conf) if save_conf else (cls, *xywh)
                                )  # label format

                                xmin = int(xyxy[0].item())
                                ymin = int(xyxy[1].item())
                                xmax = int(xyxy[2].item())
                                ymax = int(xyxy[3].item())
                                if video:
                                    filename = p.stem + "_" + str(frame)
                                else:
                                    filename = p.stem + "_" + str(count)
                                new_img = im0[ymin:ymax, xmin:xmax]
                                cv2.imwrite(
                                    str(save_dir) + "/" + filename + ".jpg", new_img
                                )
                                d_images[filename] = new_img
                                category = int(cls.item())

                                di[f"{filename}"] = dict()
                                di[f"{filename}"]["bbox_coordinates"] = [
                                    xmin,
                                    ymin,
                                    xmax,
                                    ymax,
                                ]
                                di[f"{filename}"]["confidence_score"] = conf.item()

                                if category == 0:
                                    di[f"{filename}"]["category"] = "cat"
                                else:
                                    di[f"{filename}"]["category"] = "dog"

                                di[f"{filename}"]["source"] = p.stem

                            count += 1
                        d[
                            (
                                sorted(
                                    di.items(),
                                    key=lambda x: x[1]["confidence_score"],
                                    reverse=True,
                                )[0]
                            )[0]
                        ] = (
                            sorted(
                                di.items(),
                                key=lambda x: x[1]["confidence_score"],
                                reverse=True,
                            )[0]
                        )[
                            1
                        ]

                    print(f"{s}Done. ({t2 - t1:.3f}s)")

                    # Stream results
                    # im0 = annotator.result()
                if len(d):
                    if save_txt:  # Write to file
                        file_path = project_ + "/predictions.json"
                        with open(file_path, "w") as json_file:
                            json.dump(d, json_file)
            t4 = time.time()
            print(f"completed (Detection and saved). ({t4 - t5:.3f}s)")

            if len(d):
                pred_df = (
                    pd.DataFrame(d)
                    .transpose()
                    .reset_index()
                    .rename(columns={"index": "image_name"})
                )

                pred_df_sorted = pred_df.sort_values(
                    by=["source", "confidence_score"], ascending=[True, False]
                )  # [pred_df["category"] == "cat"]
                # .head(top_frames)
                # .reset_index(drop=True)
                df_top_frames = pd.DataFrame(columns=pred_df_sorted.columns)
                for sorc in pred_df_sorted["source"].unique():
                    top_frams = pred_df_sorted[pred_df_sorted["source"] == sorc].head(
                        top_frames
                    )
                    df_top_frames = pd.concat(
                        [df_top_frames, top_frams], ignore_index=True
                    )

                save_dir = increment_path(
                    Path(project_) / "top_frames", exist_ok=exist_ok
                )  # increment run
                (save_dir).mkdir(parents=True, exist_ok=True)

                for i in df_top_frames["image_name"].values:
                    cv2.imwrite(str(save_dir) + "/" + i + ".jpg", d_images[i])

                df_top_frames.to_excel(str(project_) + f"/top_predictions.xlsx")
                print(f"Top frames saved. ({time.time() - t4:.3f}s)")
            print(f"Done. ({time.time() - t0:.3f}s)")
