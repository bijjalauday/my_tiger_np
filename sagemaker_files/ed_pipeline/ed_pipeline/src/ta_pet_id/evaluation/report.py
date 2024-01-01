"""Pipeline evaluation module.
"""

import pandas as pd
from sklearn.metrics import confusion_matrix as _confusion_matrix
from sklearn.metrics import classification_report as _classification_report

from ta_pet_id.data_prep.db_utils import write_pipeline_report


def _get_confusion_matrix_df(df, actual_col, pred_col):
    cfm = _confusion_matrix(df[actual_col], df[pred_col])
    cfm_df = pd.DataFrame({"predicted_cat": cfm[:, 0], "predicted_dog": cfm[:, 1], "predicted_others": cfm[:, 2]}, )
    cfm_df.index = ['actual_cat', 'actual_dog', 'actual_others']
    return cfm_df


def _get_classification_report(df, actual_col, pred_col, only_overall=False):
    report_list = []
    report_indices = []
    segements = ['overall', 'cat', 'dog'] if not only_overall else ['overall']
    for class_type in segements:
        report_dict = {}
        report_dict['class_type'] = class_type
        if class_type == 'overall':
            df_2 = df
        else:
            df_2 = df[df['pet_type'] == class_type]
        df_2['pet_id'] = df_2['pet_id'].astype(str)
        pet_id_count = len(df_2.groupby(['house_id','pet_id']))
        house_id_count = df_2['house_id'].nunique()
        _report = _classification_report(df_2[actual_col], df_2[pred_col], output_dict=True)
        report_dict = {key: value for key, value in _report.items() if key in ['weighted avg']}['weighted avg']
        report_dict['accuracy'] = _report['accuracy']
        report_dict['pet_id_count'] = pet_id_count
        report_dict['house_id_count'] = house_id_count
        report_indices.append(class_type)
        report_list.append(report_dict)
    cls_report_df = pd.DataFrame(report_list)
    cls_report_df.index = report_indices

    return cls_report_df


def _get_cls_report_by_segment(df, actual_col, pred_col, by_segment):
    by_segment = [by_segment] if isinstance(by_segment, str) else by_segment
    report_dfs = []
    for name, group in df.groupby(by_segment):
        group_report_df = _get_classification_report(group, actual_col, pred_col, only_overall=True)
        group_report_df.index = [name]
        group_report_df.index.name = 'group'
        report_dfs.append(group_report_df)
    report_df = pd.concat(report_dfs).reset_index()
    if len(by_segment) > 1:
        report_df[by_segment] = pd.DataFrame(report_df['group'].tolist(), index=report_df.index)
        report_df = report_df.drop(columns=['group'])
    else:
        report_df = report_df.rename(columns={'group': by_segment[0]})
    return report_df


def get_pipeline_eval_report(inference_df, return_report=True, write_report=True, report_path=""):
    """
    Generates pipeline model evaluation report for pet type and id classifications

    Parameters
    ----------
    inference_df : pd.DataFrame
      inference image metadata with predicted pet type and pet id
    return_report: bool, default=True
        whether to return report dictionary or not
    write_report: bool, default=True
        whether to write report or not
    report_path: str
        report file path and name e.g. xyz/report.xlsx

    Returns
    -------
    report_dict : dict
        returns report dictionary if return_report sets to True
    """
    report_dict = {}
    report_dict['pet_type_cls'] = {}
    report_dict['pet_id_cls'] = {}

    detect_count_df = inference_df['pred_pet_type'].map({"cat": 'Single Face Detection',
                                                         "dog": 'Single Face Detection',
                                                         "multi-face": 'Multi-Face Detection',
                                                         "other": 'No Face Detection'}).value_counts().rename_axis(
        'Detection Type').reset_index(name='#Images')
    report_dict['pet_type_cls']['detect_count'] = detect_count_df

    inference_df2 = inference_df[inference_df['pred_pet_type'] != 'multi-face'].reset_index(drop=True)
    pet_type_cls_report_df = _get_classification_report(inference_df2, 'pet_type', 'pred_pet_type')
    report_dict['pet_type_cls']['pet_type_cls_report'] = pet_type_cls_report_df
    pet_type_cfm_matrix_df = _get_confusion_matrix_df(inference_df2, 'pet_type', 'pred_pet_type')
    report_dict['pet_type_cls']['pet_type_cfm_matrix'] = pet_type_cfm_matrix_df

    inference_df3 = inference_df2[inference_df2['pred_pet_type'] != 'other'].reset_index(drop=True)
    pet_id_cls_report_df = _get_classification_report(inference_df3, 'pet_id', 'pred_pet_id')
    report_dict['pet_id_cls']['pet_id_cls_report'] = pet_id_cls_report_df

    if write_report:
        write_pipeline_report(report_dict, report_path)

    if return_report:
        return report_dict
