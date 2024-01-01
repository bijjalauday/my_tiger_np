"""
This module provides main utilities for enrollment, inference process for a household and global training process.
"""

import cv2
import numpy as np
import os
import pandas as pd
from datetime import datetime as _datetime

from ta_pet_id.pipeline import efficientnet, matcher, yolo


def enroll_pet(
    context, house_id, house_df, efficientnet_model=None, yolo_model_objs=None
):
    """
    Enroll pets of a household based on the given images of each pet

    Parameters
    ----------
    context : ta_pet_id's context object
        this contains configuration about default path of image data
        and path to which data will be written to
    house_id : str
        a unique device id attached to the product
    house_df : pd.DataFrame
        a dataframe having enroll metadata
    efficientnet_model: object
        loaded model object of trained EfficientNetB2
    yolo_model_objs: list
        loaded model and device type object of trained YOLOv5

    Examples
    --------
    >>> import os
    >>> from ta_pet_id.core.context import Context
    >>> from ta_pet_id.pipeline.main import enroll_pet
    >>> from ta_pet_id.data_prep.core import prep_enroll_data
    >>> cfg = os.environ.get("TA_LIB_APP_CONFIG_PATH")
    >>> context_obj = Context.from_config_file(cfg)
    >>> house_df = prep_enroll_data(context_obj)
    >>> enroll_pet(context_obj, house_id, house_df)
    """

    enroll_dir_path = context.data_catalog["enroll"]["raw_folder"]
    house_df = house_df[house_df["house_id"] == house_id]

    pet_info = []
    for pet_id in house_df["pet_id"].unique():
        house_pet_id = f"{house_id}-{pet_id}"
        pet_type = house_df[house_df["pet_id"] == pet_id]["pet_type"].unique()[0]
        img_paths = house_df[house_df["pet_id"] == pet_id]["image_path"].values
        imgs_count = len(img_paths)
        pet_imgs = [
            cv2.imread(os.path.join(enroll_dir_path, img_path))
            for img_path in img_paths
        ]

        (
            pred_pet_classes,
            pred_pet_coords,
            pred_conf_scores,
            pred_pet_faces,
        ) = yolo.predict(
            context, pet_imgs, yolo_model_objs
        )  # added confidence score and multi face detections

        pred_pet_valid_faces = list(filter(lambda x: x is not None, pred_pet_faces))

        min_imgs_required = context.data_catalog["enroll"]["min_imgs_required"]
        # if len(pred_pet_valid_faces) < min_imgs_required:
        #     raise ValueError(
        #         f"For House ID:{house_id} | Pet ID:{pet_id}\n"
        #         f"Face detected for {len(pred_pet_valid_faces)} images out of given {imgs_count} images!\n"
        #         f"Please provide at least {min_imgs_required - len(pred_pet_valid_faces)} more front "
        #         f"facing images."
        #     )

        pred_pet_embds = efficientnet.predict(pet_imgs, efficientnet_model)
        pred_pet_valid_embds = np.array(
            [emb for emb in pred_pet_embds if emb is not None]
        )
        pet_mean_embed = str(list(pred_pet_valid_embds.mean(axis=0)))

        current_datetime = _datetime.now()
        pet_info.append(
            [
                current_datetime.strftime("%d-%m-%Y %H:%M:%S"),
                house_id,
                pet_id,
                house_pet_id,
                pet_type,
                len(pred_pet_embds),
                len(pred_pet_valid_embds),
                pet_mean_embed,
            ]
        )

    pet_db_cols = [
        "datetime",
        "house_id",
        "pet_id",
        "house_pet_id",
        "pet_type",
        "#input_imgs",
        "#imgs_used_for_enroll",
        "embedding",
    ]
    house_pet_db = pd.DataFrame(pet_info, columns=pet_db_cols)

    return house_pet_db


def infer_pet_id(
    context,
    house_id,
    house_df,
    enrolled_pets_db,
    efficientnet_model=None,
    yolo_model_objs=None,
):
    """
    Infer pet IDs for given pet images of a household

    Parameters
    ----------
    context : ta_pet_id's context object
        this contains configuration about default path of image data
        and path to which data will be written to
    house_id : str
        house name or device ID of camera
    house_df : pd.DataFrame
        a dataframe having enroll metadata
    enrolled_pets_db: pd.DataFrame
        enrolled pets db
    efficientnet_model: object
        loaded model object of trained EfficientNetB2
    yolo_model_objs: list
        loaded model and device type object of trained YOLOv5

    Returns
    -------
    prediction : pd.DataFrame
        prediction dictionary contains predicted IDs, detection and recognition evaluation report.

    Examples
    --------
    >>> import os
    >>> from ta_pet_id.core.context import Context
    >>> from ta_pet_id.pipeline.main import infer_pet_id
    >>> from ta_pet_id.data_prep.core import prep_inference_data
    >>> cfg = os.environ.get("TA_LIB_APP_CONFIG_PATH")
    >>> context_obj = Context.from_config_file(cfg)
    >>> house_df = prep_inference_data(context_obj)
    >>> preds = infer_pet_id(context_obj, house_id, house_df)
    >>> print(f"predicted pet ids are: {preds['pred_ids']}")

    """

    infer_dir_path = context.data_catalog["inference"]["raw_folder"]
    house_df = house_df[house_df["house_id"] == house_id]
    pet_imgs_paths = house_df["image_path"].values
    PET_IMGS = [
        cv2.imread(os.path.join(infer_dir_path, _path)) for _path in pet_imgs_paths
    ]
    print(len(PET_IMGS))
    pred_pet_types, pred_pet_coords, pred_conf_scores, pred_pet_faces = yolo.predict(
        context, PET_IMGS, yolo_model_objs
    )  # ( added confidence score and multi face detections))
    # print(pred_pet_types,pred_pet_types,pred_pet_coords,pred_conf_scores,pred_pet_faces)
    pred_pet_embds = efficientnet.predict(PET_IMGS, efficientnet_model)
    enrolled_hh_pets_db = enrolled_pets_db[enrolled_pets_db["house_id"] == house_id]
    pred_pet_ids = matcher.get_pet_id(
        pred_pet_embds, pred_pet_types, enrolled_hh_pets_db
    )
    # print(pred_pet_ids)
    house_df["pred_pet_type"] = pred_pet_types
    house_df["pred_pet_id"] = pred_pet_ids
    # house_df['confidence_score'] = pred_conf_scores

    # house_df = house_df.explode('pred_pet_id')

    return house_df


def train_efficientnet(context, train_meta_df):
    """
    Train the EfficientNetB2 model

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object
    train_meta_df : pandas.DataFrame
        training images metadata
    """
    efficientnet.train(context, train_meta_df)


def train_yolo(context, train_meta_df):
    """
    Train the YOLOv5 model

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object
    train_meta_df : pandas.DataFrame
        training images metadata
    """
    yolo.train(context, train_meta_df)
