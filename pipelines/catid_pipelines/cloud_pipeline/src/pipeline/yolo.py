"""
This module provides the core and helper utilities for setup, prediction and training of YOLOv5 model.
"""
import os
import pathlib

from ta_pet_id.core.utils import get_package_path as _get_package_path
from ta_pet_id.data_prep import db_utils
from ta_pet_id.yolo import data_prep
from ta_pet_id.yolo.detect import yolo_detect as _yolo_detect
from ta_pet_id.yolo.models.common import (
    DetectMultiBackend as _DetectMultiBackend,
)
from ta_pet_id.yolo.train import run as _run
from ta_pet_id.yolo.utils.torch_utils import select_device as _select_device


def load_model(context):
    """
    Load trained YOLOv5 model

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object

    Returns
    -------
    model_objs : list
      list having YOLOv5 model and device type object

    """

    weights = context.yolo["scoring"]["model_path"].format(**context.yolo["scoring"])
    device = context.yolo["scoring"]["device"].format(**context.yolo["scoring"])

    device = _select_device(device)
    model = _DetectMultiBackend(weights, device=device, dnn=False)
    model_objs = [model, device]
    return model_objs


def predict(context, img_array, model_objs):
    """
    Detects pet's (cat and dog) faces in the given images.

    Given a list of images it outputs the list of pet classes ,list of bounding box co-ordinates
    and list of cropped faces from yolo model.
    The above predictions are returned for an image only if single face was detected,
    else None is returned for that image.

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object
    imgs :list
      list of images(np.array) in BGR format
    model_objs : list
      list having YOLOv5 model and device type object

    Returns
    -------
    output : list
        list of list of pet types, list of bounding box co-ordinates , confidence score and list of cropped faces.

    """

    conf_thresh = context.yolo["scoring"]["conf_threshold"]
    iou_thresh = context.yolo["scoring"]["iou_threshold"]

    model, device = model_objs

    output = _yolo_detect(
        model,
        device,
        img_array,
        imgsz=[256, 256],
        conf_thres=conf_thresh,
        iou_thres=iou_thresh,
    )

    return output


def train(context, train_meta_df):
    """
    Train the YOLOv5 model.

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object
    train_meta_df : pandas.DataFrame
        training images metadata

    """
    training_version = db_utils._training_model_version(context, "yolo")
    train_meta, val_meta = data_prep.create_train_test_for_yolo(
        context, train_meta_df, training_version
    )

    dir_path = context.data_catalog["yolo"]["raw_folder"]
    train_imgs = train_meta["image_path"].tolist()
    train_imgs = [os.path.join(dir_path, img_path) for img_path in train_imgs]
    val_imgs = val_meta["image_path"].tolist()
    val_imgs = [os.path.join(dir_path, img_path) for img_path in val_imgs]

    model_configs_path = os.path.join(
        _get_package_path(), "ta_pet_id", "yolo", "model_conf"
    )

    _run(
        train_imgs=train_imgs,
        val_imgs=val_imgs,
        weights=context.yolo["training"]["pre_trained_weights"].format(
            **context.yolo["scoring"]
        ),
        cfg=os.path.join(model_configs_path, "custom_yolov5s.yaml"),
        data=os.path.join(model_configs_path, "classes.yaml"),
        hyp=os.path.join(model_configs_path, "hyp.scratch.yaml"),
        epochs=context.yolo["training"]["epochs"],
        batch_size=context.yolo["training"]["batch_size"],
        imgsz=256,
        device=context.yolo["training"]["device"],
        rect=context.yolo["training"]["rect"],
        save_period=context.yolo["training"]["save_period"],
        patience=context.yolo["training"]["patience"],
        freeze=context.yolo["training"]["freeze"],
        save_dir=os.path.join(
            context.data_catalog["yolo"]["artifacts_folder"], training_version, "model"
        ),
        local_rank=context.yolo["training"]["local_rank"],
        resume=context.yolo["training"]["resume"],
        nosave=context.yolo["training"]["nosave"],
        noval=context.yolo["training"]["noval"],
        noautoanchor=context.yolo["training"]["noautoanchor"],
        cache=context.yolo["training"]["cache"],
        image_weights=context.yolo["training"]["image_weights"],
        multi_scale=context.yolo["training"]["multi_scale"],
        single_cls=context.yolo["training"]["single_cls"],
        adam=context.yolo["training"]["adam"],
        sync_bn=context.yolo["training"]["sync_bn"],
        workers=context.yolo["training"]["workers"],
        quad=context.yolo["training"]["quad"],
        linear_lr=context.yolo["training"]["linear_lr"],
        label_smoothing=context.yolo["training"]["label_smoothing"],
        entity=context.yolo["training"]["entity"],
        upload_dataset=context.yolo["training"]["upload_dataset"],
        bbox_interval=context.yolo["training"]["bbox_interval"],
        artifact_alias=context.yolo["training"]["artifact_alias"],
    )
