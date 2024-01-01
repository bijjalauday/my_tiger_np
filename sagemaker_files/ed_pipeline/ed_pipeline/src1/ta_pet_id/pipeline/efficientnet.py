"""
This module provides the core and helper utilities for setup, prediction and training of EfficientNetB2 model.
"""

import cv2
import numpy as np
import os
import tensorflow as tf

from ta_pet_id.data_prep import db_utils
from ta_pet_id.efficientnet import dataloader, siamese_model
from ta_pet_id.pipeline import yolo

os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
mirrored_strategy = tf.distribute.MirroredStrategy(
    cross_device_ops=tf.distribute.HierarchicalCopyAllReduce()
)


def load_model(context):
    """
    Load trained EfficientNetB2 model

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object

    Returns
    -------
    model : keras.engine.functional.Functional
        EfficientNetB2 model object

    """
    path = context.efficientnet["scoring"]["model_path"].format(
        **context.efficientnet["scoring"]
    )
    with mirrored_strategy.scope():
        model = tf.keras.models.load_model(path)
    return model


def predict(imgs, model):
    """
    Generates feature vector(embedding) for the given images.

    Parameters
    ----------
    imgs : list
      list of images(np.array) in BGR format
    model : object
        EfficientNetB2 model object (loaded from load_model function)

    Returns
    -------
    embeddings : np.array
        list of feature vector(embeddings)
    """

    embeddings = []
    for img in imgs:
        if img is None:
            embeddings.append(None)

        # elif (type(img) == list) & (
        #     len(img) > 1
        # ):  # For multiple predictions the type will be list
        #     embed = []
        #     for im in img:
        #         RGB_img = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        #         resized_img = cv2.resize(RGB_img, (256, 256))
        #         embedding = model.predict(np.expand_dims(resized_img, axis=0))
        #         embed.append(embedding[0])
        #     embeddings.append(embed)
        else:
            RGB_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            resized_img = cv2.resize(RGB_img, (256, 256))
            embedding = model.predict(np.expand_dims(resized_img, axis=0))
            embeddings.append(embedding[0])
    return np.array(embeddings, dtype=object)


def train(context, train_meta_df):
    """
    Train the EfficientNetB2 model

    Parameters
    ----------
    context : ta_pet_id's Context object
        project configuration's object
    train_meta_df : pandas.DataFrame
        training images metadata
    """
    version = db_utils._training_model_version(context, "efficientnet")
    # get the YOLO prediction on the images
    base_path = context.data_catalog["efficientnet"]["raw_folder"]
    pet_imgs = [
        cv2.imread(os.path.join(base_path, img_path))
        for img_path in train_meta_df["image_path"].values
    ]
    model, device = yolo.load_model(context)
    train_meta_df["face_loc"] = yolo.predict(context, pet_imgs, [model, device])[1]

    filter_train_df = dataloader.process_train_data(context, train_meta_df)
    db_utils.save_efficientnet_processed_train_metadata_db(
        context, filter_train_df, version
    )

    train_dataset, test_dataset = dataloader.train_test_split(context, filter_train_df)
    model_path = os.path.join(
        os.path.join(
            context.data_catalog["efficientnet"]["artifacts_folder"], version, "model"
        )
    )
    if not os.path.exists(model_path):
        os.makedirs(model_path)

    class MyCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            """
            Function to save a model at any epoch

            Parameters
            ----------
            epoch : int
                epoch at which model will get saved
            """

            ALERADY_TRAINED_EPOCHS = str(epoch)
            backbone_model.save(model_path + "/" + ALERADY_TRAINED_EPOCHS + "_back")

    IMG_SIZE = 256
    with mirrored_strategy.scope():
        pretrained = context.efficientnet["training"]["pre_trained_weights"].format(
            **context.efficientnet["scoring"]
        )
        if pretrained == "":
            backbone_model = siamese_model.siamese_backbone(IMG_SIZE)
        else:
            backbone_model = tf.keras.models.load_model(pretrained)
        training_model = siamese_model.siamese_network(backbone_model, IMG_SIZE)
        final_model = siamese_model.TripletLossModel(training_model)
        final_model.compile(
            optimizer=tf.keras.optimizers.Adam(
                context.efficientnet["training"]["learning_rate"]
            )
        )
    final_model.fit(
        train_dataset,
        validation_data=test_dataset,
        epochs=context.efficientnet["training"]["epoch"],
        verbose=context.efficientnet["training"]["verbose"],
        callbacks=[MyCallback()],
    )
