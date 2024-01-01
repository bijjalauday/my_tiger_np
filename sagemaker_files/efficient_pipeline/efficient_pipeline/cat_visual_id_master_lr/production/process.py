"""
This is core production module to run the processes globally i.e. across all the households.

This provides following functionalities:
    - Run enrollment process
    - Run inference process
    - Run training of EfficientNetB2 model
    - Run training of YOLOv5 model
"""
import pandas as pd
from ta_pet_id.core.context import create_context as _create_context
from ta_pet_id.pipeline.main import (enroll_pet as _enroll_pet,
                                     infer_pet_id as _infer_pet_id,
                                     train_efficientnet as _train_efficientnet,
                                     train_yolo as _train_yolo)
from ta_pet_id.data_prep import db_utils
from ta_pet_id.data_prep import core
from ta_pet_id.pipeline import yolo, efficientnet
from ta_pet_id.evaluation.report import get_pipeline_eval_report as _get_pipeline_eval_report


def run_enrollment(conf_path, re_enroll=False):
    """Run the enrollment process on given raw pet images data for different households.

    Parameters
    ----------
    conf_path : str
       path of the project's `conf` folder
    re_enroll : bool
      whether to re-enroll the existing enrolled pets or not
      Note: To overwrite existing enrollment, set it to True

    """

    # create the project configuration object
    context_obj = _create_context(conf_path)

    # get list of existing enrolled pets
    enrolled_pets = []
    append_new_pets = False
    if not re_enroll:
        enrolled_pets = db_utils.load_enrolled_pet_db(context_obj)['house_pet_id'].unique()
        append_new_pets = True

    # prepare enroll metadata from the raw image data
    enroll_meta_df = core.prep_enroll_data(context_obj, enrolled_pets)
    if len(enroll_meta_df) == 0:
        raise Warning(f"All the pets from the given enroll data are already enrolled! If you want to re-enroll, "
                      f"run the process again with re_enroll=True.")
    # save enroll metadata to the DB
    db_utils.save_enroll_metadata_db(context_obj, enroll_meta_df, append=append_new_pets)

    # load the trained pipeline models
    eff_net_model = efficientnet.load_model(context_obj)
    yolo_model_objs = yolo.load_model(context_obj)

    # execute enroll process for each house
    house_dfs = []
    for house_id in enroll_meta_df['house_id'].unique():
        house_df = enroll_meta_df[enroll_meta_df['house_id'] == house_id]
        house_pet_db = _enroll_pet(context_obj, house_id, house_df, eff_net_model, yolo_model_objs)
        house_dfs.append(house_pet_db)

    # consolidate households pets into one and save the enrolled pet info to DB
    enrolled_pet_df = pd.concat(house_dfs, ignore_index=True)
    db_utils.save_enrolled_pet_db(context_obj, enrolled_pet_df, append=append_new_pets)
    print("Pet enrollment completed for all houses!")


def run_inference(conf_path, return_eval_report=False):
    """Run the inference process on given raw pet images data for different households.

    Parameters
    ----------
    conf_path : str
       path of the project's `conf` folder
    return_eval_report : bool
      whether to get classification evaluation report or not
      Note: Subject to if truth label instances are provided or not

    Returns
    -------
    report: dict or None
        a report dict if truth labels are provided otherwise None

    """

    # create the project configuration object
    context_obj = _create_context(conf_path)

    # prepare inference metadata from the raw image data
    infer_meta_df = core.prep_inference_data(context_obj)
    # save inference metadata to the DB
    db_utils.save_inference_metadata_db(context_obj, infer_meta_df)

    # load the trained pipeline models
    eff_net_model = efficientnet.load_model(context_obj)
    yolo_model_objs = yolo.load_model(context_obj)

    # load enrolled pets db
    enrolled_pets_db = db_utils.load_enrolled_pet_db(context_obj)
    enrolled_houses = enrolled_pets_db['house_id'].unique()

    # execute inference process for each house
    house_dfs = []
    for house_id in infer_meta_df['house_id'].unique():
        if house_id not in enrolled_houses:
            # as no pets are enrolled from this house, inference can't be given for the same
            continue
        house_df = infer_meta_df[infer_meta_df['house_id'] == house_id]
        house_df_preds = _infer_pet_id(context_obj, house_id, house_df, enrolled_pets_db, eff_net_model,
                                       yolo_model_objs)
        house_dfs.append(house_df_preds)

    # consolidate households pets into one and save the inference data predictions to DB
    inference_preds_df = pd.concat(house_dfs, ignore_index=True)
    db_utils.save_inference_prediction_db(context_obj, inference_preds_df)

    # select the inference data for which truth label are given and generate evaluation report
    eval_df = inference_preds_df[inference_preds_df['pet_type'].notnull() &
                                 inference_preds_df['pet_id'].notnull()]
    if len(eval_df) > 0:
        report_path = context_obj.data_catalog['inference']['inference_preds_report'].format(
            **context_obj.data_catalog['inference'])
        report = _get_pipeline_eval_report(eval_df, return_report=return_eval_report, write_report=True,
                                           report_path=report_path)
    if return_eval_report:
        return report


def train_efficientnet_model(conf_path):
    """Run the EfficientNetB2 training process on given raw pet images data for different households.

    Parameters
    ----------
    conf_path : str
       path of the project's `conf` folder

    """

    # create the project configuration object
    context_obj = _create_context(conf_path)

    # prepare train meta data from the raw image data
    train_meta_df = core.prep_efficientnet_train_metadata(context_obj)
    db_utils.save_efficientnet_train_metadata_db(context_obj, train_meta_df)

    # run efficientnet training process
    _train_efficientnet(context_obj, train_meta_df)


def train_yolo_model(conf_path):
    """Run the YOLOv5 training process on given raw pet images data for different households.

    Parameters
    ----------
    conf_path : str
       path of the project's `conf` folder

    """

    # create the project configuration object
    context_obj = _create_context(conf_path)

    # prepare train meta data from the raw image data
    train_meta_df = core.prep_yolo_train_metadata(context_obj)
    db_utils.save_yolo_train_metadata_db(context_obj, train_meta_df)

    # run yolo training process
    _train_yolo(context_obj, train_meta_df)
