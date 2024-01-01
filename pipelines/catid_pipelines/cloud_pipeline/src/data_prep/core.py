"""
Module contains utilities to prepare metadata from the given raw images data for enroll, inference and training processes.
"""
import os
import re
import shutil
from datetime import datetime as _datetime

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm as _tqdm

from src.data_prep import db_utils


def prep_enroll_data(context, enrolled_pets=None, augment=True):
    """
    Prepare a database of raw enroll image dataset

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object
    enrolled_pets: list
        list of existing enrolled pets ids in format of "<house_id>-<pet_id>"

    Returns
    -------
    df_new : pandas.DataFrame
        prepared enroll metadata
    """
    if enrolled_pets is None:
        enrolled_pets = []
    image_info = []
    dir_path = context.data_catalog["enroll"]["raw_folder"]
    for house_name in _tqdm(os.listdir(rf"{dir_path}")):
        if not os.path.isdir(os.path.join(dir_path, house_name)):
            continue
        for pet_type in os.listdir(os.path.join(dir_path, house_name)):
            for pet in os.listdir(os.path.join(dir_path, house_name, pet_type)):
                if not os.path.isdir(os.path.join(dir_path, house_name, pet_type, pet)):
                    continue
                house_pet_id = f"{house_name}-{pet}"
                if house_pet_id in enrolled_pets:
                    continue
                if augment:
                    file_nam = "enroll_with_aug"
                else:
                    file_nam = "enroll_images"
                for pet_image in os.listdir(
                    os.path.join(dir_path, house_name, pet_type, pet, file_nam)
                ):
                    img_path = os.path.join(
                        house_name, pet_type, pet, file_nam, pet_image
                    )
                    img = cv2.imread(os.path.join(dir_path, img_path))
                    if img is None:
                        continue
                    image_info.append(
                        [
                            _datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                            house_name,
                            pet,
                            house_pet_id,
                            pet_type,
                            img_path,
                        ]
                    )

    meta_df_cols = [
        "datetime",
        "house_id",
        "pet_id",
        "house_pet_id",
        "pet_type",
        "image_path",
    ]
    df_new = pd.DataFrame(image_info, columns=meta_df_cols)

    return df_new


def prep_inference_data(context):
    """
    prepare inference metadata

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object

    Returns
    -------
    df_new : pandas.DataFrame
        prepared inference metadata.

    """

    image_info = []
    dir_path = context.data_catalog["inference"]["raw_folder"]

    inference_labels_df = db_utils.load_inference_label_db(context)

    for house_name in _tqdm(os.listdir(rf"{dir_path}")):
        if not os.path.isdir(os.path.join(dir_path, house_name)):
            continue
        top_frames = "top_frames"
        excel_path = os.path.join(dir_path, house_name, "top_predictions.xlsx")
        pred_exl = pd.read_excel(excel_path)
        for pet_image in os.listdir(os.path.join(dir_path, house_name, top_frames)):
            pet_im_ind = pet_image.rfind(".")
            pet_im = pet_image[:pet_im_ind]
            pet_catg = pred_exl[pred_exl["image_name"] == pet_im]["category"].values[0]
            img_path = os.path.join(house_name, top_frames, pet_image)
            img = cv2.imread(os.path.join(dir_path, img_path))
            if img is None:
                continue
            image_info.append(
                [
                    _datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    house_name,
                    pet_catg,
                    img_path,
                ]
            )

    df_new = pd.DataFrame(
        image_info, columns=["datetime", "house_id", "category", "image_path"]
    )
    if len(inference_labels_df) > 0:
        df_new = df_new.merge(
            inference_labels_df, how="left", on=["house_id", "category", "image_path"]
        )
    else:
        df_new["pet_type"] = None
        df_new["pet_id"] = None

    return df_new


def prep_efficientnet_train_metadata(context):
    """
    generate EfficientNetB2 train metadata

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object

    Returns
    -------
    pandas.DataFrame
        prepared metadata as pandas DataFrame object

    """
    image_info = []
    dir_path = context.data_catalog["efficientnet"]["raw_folder"]
    for house_name in _tqdm(os.listdir(dir_path)):
        if not os.path.isdir(os.path.join(dir_path, house_name)):
            continue
        for pet_type in os.listdir(os.path.join(dir_path, house_name)):
            for pet in os.listdir(os.path.join(dir_path, house_name, pet_type)):
                if not os.path.isdir(os.path.join(dir_path, house_name, pet_type, pet)):
                    continue
                house_pet_id = f"{house_name}-{pet}"
                for pet_image in os.listdir(
                    os.path.join(dir_path, house_name, pet_type, pet)
                ):
                    img_path = os.path.join(
                        dir_path, house_name, pet_type, pet, pet_image
                    )
                    img = cv2.imread(img_path)
                    if img is None:
                        continue
                    image_info.append(
                        [
                            _datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                            house_name,
                            pet,
                            house_pet_id,
                            pet_type,
                            os.path.join(house_name, pet_type, pet, pet_image),
                        ]
                    )

    meta_df_cols = [
        "datetime",
        "house_id",
        "pet_id",
        "house_pet_id",
        "pet_type",
        "image_path",
    ]
    df_new = pd.DataFrame(image_info, columns=meta_df_cols)

    return df_new


def prep_yolo_train_metadata(context):
    """
    generate YOLOv5 train metadata

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object

    Returns
    -------
    pandas.DataFrame
        prepared metadata as pandas DataFrame object

    """

    dir_path = context.data_catalog["yolo"]["raw_folder"]

    info = []
    for img in os.listdir(os.path.join(dir_path, "images")):
        img_path = os.path.join(os.path.join(dir_path, "images", img))
        img_ = cv2.imread(img_path)
        if img_ is None:
            continue
        img_name = img.rsplit(".", 1)[0]
        label_name_ = img_name + ".txt"
        if label_name_ in os.listdir(os.path.join(dir_path, "labels")):
            if os.stat(os.path.join(dir_path, "labels", label_name_)).st_size != 0:
                f = open(os.path.join(dir_path, "labels", label_name_), "r")
                count = 0
                for line in f:
                    count = count + 1
                    line = line.split(" ")
                    pet_type = line[0]
                    if pet_type == "0":
                        pet_type = "cat"
                    elif pet_type == "1":
                        pet_type = "dog"
                    else:
                        raise ValueError(f"not possible label for yolo class type")
                if count > 1:
                    pet_type = "multiple"
                info.append(
                    [
                        _datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                        img,
                        label_name_,
                        os.path.join("images", img),
                        pet_type,
                        1,
                    ]
                )

            else:
                info.append(
                    [
                        _datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                        img,
                        label_name_,
                        os.path.join("images", img),
                        "other",
                        1,
                    ]
                )
        else:
            info.append(
                [
                    _datetime.now().strftime("%d-%m-%Y %H:%M:%S"),
                    img,
                    np.nan,
                    os.path.join("images", img),
                    np.nan,
                    0,
                ]
            )

    meta_df_cols = [
        "datetime",
        "image_name",
        "label_name",
        "image_path",
        "pet_type",
        "to_use",
    ]

    meta_df = pd.DataFrame(info, columns=meta_df_cols)

    return meta_df


def prep_yolo_data_from_hh(context):
    """
    Utility function to convert data from household level to the format required by YOLOv5

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object

    """

    db_path = context.data_catalog["enroll"]["raw_folder"]
    out_imgs_path = os.path.join(context.data_catalog["yolo"]["raw_folder"], "images")
    out_labels_path = os.path.join(context.data_catalog["yolo"]["raw_folder"], "labels")
    os.makedirs(out_imgs_path, exist_ok=True)
    os.makedirs(out_labels_path, exist_ok=True)
    for root, dirs, files in os.walk(db_path):
        for img in files:
            img_ = cv2.imread(os.path.join(root, img))
            if img_ is None:
                continue
            img_name = img.rsplit(".", 1)[0]
            label_path_ = img_name + ".txt"
            rel_path = os.path.relpath(root, db_path)

            base_, pet_id = os.path.split(rel_path)
            base_, pet_type = os.path.split(base_)
            base_, h_id = os.path.split(base_)

            out_img_name = h_id + "_" + pet_type + "_" + pet_id + "_" + img
            if label_path_ in files:
                shutil.copy(
                    os.path.join(root, img), os.path.join(out_imgs_path, out_img_name)
                )
                shutil.copy(
                    os.path.join(root, label_path_),
                    os.path.join(
                        out_labels_path, out_img_name + out_img_name.rsplit(".", 1)[-1]
                    ),
                )
