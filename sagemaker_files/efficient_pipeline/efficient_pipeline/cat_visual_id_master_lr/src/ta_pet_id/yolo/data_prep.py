import random

from sklearn.model_selection import train_test_split
from src.ta_pet_id.data_prep import db_utils


def create_train_test_for_yolo(context,train_meta_df,training_version,seed=42):
    """
    creates yolo train_test data from the database folder

    Parameters
    ----------
    context : object
       context object to locate config parameters

    train_meta_df : pandas dataframe
     metadata dataframe used for training yolo

    training_version: string
     training version string, to keep track for train pickle files

    seed :int
     default value is 42


    Returns
    -------
    train : pandas dataframe
     meta_df of yolo train data.
    val : pandas dataframe
     meta_df of yolo val data.
    """
    random.seed(seed)

    train_meta_df = train_meta_df[train_meta_df['to_use'] == 1]
    train_meta_df = train_meta_df[train_meta_df['label_name'].notnull()]
    train_meta_df['sample_type'] = ''

    train_index, test_index = train_test_split(train_meta_df.index, test_size=0.3,
                                               stratify=train_meta_df['pet_type'])
    for i in train_index:
        train_meta_df['sample_type'].iloc[i] = 'train'
    for i in test_index:
        train_meta_df['sample_type'].iloc[i] = 'val'

    db_utils.save_yolo_processed_train_metadata_db(context, train_meta_df,training_version)

    train_df = train_meta_df[train_meta_df['sample_type'] == 'train']
    val_df = train_meta_df[train_meta_df['sample_type'] == 'val']
    return train_df, val_df

