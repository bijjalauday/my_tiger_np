import os
import pandas as pd


def _save_db(db, db_path):
    if not os.path.exists(os.path.dirname(db_path)):
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
    try:
        db.to_excel(db_path, index=False)
        return True
    except Exception as e:
        print(e)
        return False


def _load_db(db_path):
    db = pd.read_excel(db_path, engine='openpyxl')
    return db


def _load_db_lables_file(db_path):
    db = pd.read_csv(db_path, sep=',')
    return db


def load_enrolled_pet_db(context):
    db_path = context.data_catalog['enroll']['enrolled_pet_db'].format(**context.data_catalog['enroll'])
    if os.path.exists(db_path):
        db_csv = _load_db(db_path)
        db_csv['embedding'] = db_csv['embedding'].apply(eval)
    else:
        db_csv = pd.DataFrame(columns=['datetime', 'house_id', 'pet_id', 'house_pet_id', 'pet_type',
                                       '#input_imgs', '#imgs_used_for_enroll', 'embedding'])
    return db_csv


def save_enrolled_pet_db(context, data, append=True):
    db_path = context.data_catalog['enroll']['enrolled_pet_db'].format(**context.data_catalog['enroll'])
    if append:
        existing_db = load_enrolled_pet_db(context)
        data = pd.concat([existing_db, data], ignore_index=True)
    return _save_db(data, db_path)


def load_enroll_metadata_db(context):
    db_path = context.data_catalog['enroll']['raw_metadata_db'].format(**context.data_catalog['enroll'])
    if os.path.exists(db_path):
        db_csv = _load_db(db_path)
    else:
        db_csv = pd.DataFrame(columns=['datetime', 'house_id', 'pet_id', 'house_pet_id', 'pet_type', 'image_path'])
    db_csv['pet_id'] = db_csv['pet_id'].astype(str)
    return db_csv


def save_enroll_metadata_db(context, data, append=True):
    db_path = context.data_catalog['enroll']['raw_metadata_db'].format(**context.data_catalog['enroll'])
    if append:
        existing_db = load_enroll_metadata_db(context)
        data = pd.concat([existing_db, data], ignore_index=True)
    return _save_db(data, db_path)


def load_inference_metadata_db(context):
    db_path = context.data_catalog['inference']['raw_metadata_db']

    if os.path.exists(db_path):
        db_csv = _load_db(db_path)
    else:
        db_csv = pd.DataFrame(
            columns=['datetime', 'house_id', 'image_path', 'pet_type', 'pet_id', 'pred_pet_type', 'pred_pet_id'])
    return db_csv


def load_inference_label_db(context):
    db_path = context.data_catalog['inference']['inference_truth_label_db'].format(**context.data_catalog['inference'])
    if os.path.exists(db_path):
        db_csv = _load_db_lables_file(db_path)
        db_csv = db_csv[['house_id', 'pet_type', 'pet_id', 'image_path']]
    else:
        db_csv = pd.DataFrame(columns=['house_id', 'pet_type', 'pet_id', 'image_path'])
    return db_csv


def save_inference_metadata_db(context, data):
    db_path = context.data_catalog['inference']['raw_metadata_db'].format(**context.data_catalog['inference'])
    return _save_db(data, db_path)


def save_inference_prediction_db(context, data):
    db_path = context.data_catalog['inference']['inference_preds_db'].format(**context.data_catalog['inference'])
    return _save_db(data, db_path)


def _training_model_version(context, model):
    if model == "efficientnet":
        curr_versions = os.listdir(context.data_catalog['efficientnet']['artifacts_folder'])
    elif model == "yolo":
        curr_versions = os.listdir(context.data_catalog['yolo']['artifacts_folder'])
    else:
        raise ValueError(f"Unknown model type:{model}")
    curr_versions.remove('.gitkeep')
    if len(curr_versions) < 1:
        return 'v1.0'
    else:
        next_version = max([int(list(x)[1]) for x in curr_versions]) + 1
        return 'v' + str(next_version) + '.0'


def load_efficientnet_train_metadata_db(context):
    db_path = context.data_catalog['efficientnet']['raw_metadata_db'].format(**context.data_catalog['efficientnet'])
    if os.path.exists(db_path):
        db_csv = _load_db(db_path)
    else:
        db_csv = pd.DataFrame(columns=['datetime', 'house_id', 'pet_id', 'house_pet_id', 'pet_type', 'image_path'])
    return db_csv


def save_efficientnet_train_metadata_db(context, data):
    db_path = context.data_catalog['efficientnet']['raw_metadata_db'].format(**context.data_catalog['efficientnet'])
    return _save_db(data, db_path)


def load_efficientnet_processed_train_metadata_db(context):
    db_path = context.data_catalog['efficientnet']['model_metadata_db'].format(context.data_catalog['efficientnet'])
    if os.path.exists(db_path):
        db_csv = _load_db(db_path)
    else:
        db_csv = None
        print("No processed data file for the efficientnet training!")

    return db_csv


def save_efficientnet_processed_train_metadata_db(context, data, training_verison):
    db_path = context.data_catalog['efficientnet']['model_metadata_db'].format(**context.data_catalog['efficientnet'])
    db_path = os.path.join(os.path.dirname(db_path), training_verison,'train', os.path.basename(db_path))
    return _save_db(data, db_path)


def load_yolo_train_metadata_db(context):
    db_path = context.data_catalog['yolo']['raw_metadata_db']
    if os.path.exists(db_path):
        db_csv = _load_db(db_path)
    else:
        db_csv = pd.DataFrame(columns=['datetime', 'house_id', 'pet_type', 'pet_id', 'image_path'])
    return db_csv


def save_yolo_train_metadata_db(context, data):
    db_path = context.data_catalog['yolo']['raw_metadata_db'].format(**context.data_catalog['yolo'])
    return _save_db(data, db_path)


def load_yolo_processed_train_metadata_db(context):
    db_path = context.data_catalog['yolo']['model_metadata_db'].format(**context.data_catalog['yolo'])
    if os.path.exists(db_path):
        db_csv = _load_db(db_path)
    else:
        db_csv = None
        print("No processed data file for the yolo training!")
    return db_csv


def save_yolo_processed_train_metadata_db(context, data, training_verison):
    db_path = context.data_catalog['yolo']['model_metadata_db'].format(**context.data_catalog['yolo'])
    db_path = os.path.join(os.path.dirname(db_path), training_verison,'train', os.path.basename(db_path))
    return _save_db(data, db_path)


def write_pipeline_report(report, report_path=""):
    writer = pd.ExcelWriter(report_path, engine='xlsxwriter')

    pet_type_report = report['pet_type_cls']
    detect_count_df = pet_type_report['detect_count']
    yolo_cls_report_df = pet_type_report['pet_type_cls_report']
    yolo_cfm_matrix_df = pet_type_report['pet_type_cfm_matrix']
    detect_count_df.to_excel(writer, index=False, sheet_name=f'pets_type_report')
    yolo_cls_report_df.to_excel(writer, startrow=detect_count_df.shape[0] + 3, index=True,
                                sheet_name=f'pets_type_report')
    yolo_cfm_matrix_df.to_excel(writer, startrow=detect_count_df.shape[0] + 3, startcol=yolo_cls_report_df.shape[1] + 2,
                                index=True, sheet_name=f'pets_type_report')

    pet_id_cls_report = report['pet_id_cls']['pet_id_cls_report']
    pet_id_cls_report.to_excel(writer, index=True, sheet_name='pet_id_cls_report')

    writer.save()

    print(f"Pipeline performance report is saved at: {report_path}.")
