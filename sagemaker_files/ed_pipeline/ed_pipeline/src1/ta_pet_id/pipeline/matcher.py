# """
# This module contains core utilities used for matching embeddings and getting predicted pet IDs
# """

import numpy as np

"""
This module contains core utilities used for matching embeddings and getting predicted pet IDs
"""

import numpy as np


def get_pet_id(pred_pet_embds, pred_pet_types, enrolled_hh_pets_db):
    """
    match list of embeddings with stored embeddings in database

    Parameters
    ----------
    context : ta_pet_id's context object
        this contains path to DB
    pred_pet_embds : list
        list of embeddings
    pred_pet_types :list
        list of pet types with same length as of pred_pet_embds
    enrolled_hh_pets_db : pd.DataFrame
        enrolled pets db for a particular household

    Returns
    -------
    list
        list of predicted pet ids
    """
    res = []
    for pred_embed, pred_pet_type in zip(pred_pet_embds, pred_pet_types):
        if pred_embed is None or pred_pet_type is None:
            res.append(None)
            continue

        # house_sub_db = enrolled_hh_pets_db[enrolled_hh_pets_db['pet_type'] == pred_pet_type]
        # if len(house_sub_db) == 0:
        #     # Assuming all pets are enrolled, YOLO prediction of pet type could be wrong.
        #     house_sub_db = enrolled_hh_pets_db.copy()
        house_sub_db = enrolled_hh_pets_db.copy()
        min_dist = 99999
        pred_pet_id = None
        for i in range(len(house_sub_db)):
            row = house_sub_db.iloc[i]
            dist = np.sum(np.square(np.subtract(row['embedding'], pred_embed)))
            if dist < min_dist:
                min_dist = dist
                pred_pet_id = row['pet_id']
        res.append(pred_pet_id)
    return res



# def get_pet_id(pred_pet_embds, pred_pet_types, enrolled_hh_pets_db):
#     """
#     match list of embeddings with stored embeddings in database

#     Parameters
#     ----------
#     context : ta_pet_id's context object
#         this contains path to DB
#     pred_pet_embds : list
#         list of embeddings
#     pred_pet_types :list
#         list of pet types with same length as of pred_pet_embds
#     enrolled_hh_pets_db : pd.DataFrame
#         enrolled pets db for a particular household

#     Returns
#     -------
#     list
#         list of predicted pet ids
#     """
#     res = []
#     for pred_embed, pred_pet_type in zip(pred_pet_embds, pred_pet_types):
#         if pred_embed is None or pred_pet_type is None:
#             res.append(None)
#             continue

#         elif (
#             type(pred_embed) == list or type(pred_pet_type) == list
#         ):  # If there is any multi detection it type will be list
#             pred_pet_id = []
#             d = {}
#             ids = []
#             # for pred_em, pred_pet_ty in zip(pred_embed,pred_pet_type):

#             for rows in range(0, len(pred_embed)):
#                 house_sub_db = enrolled_hh_pets_db[
#                     enrolled_hh_pets_db["pet_type"] == pred_pet_type[rows]
#                 ]

#                 if len(house_sub_db) == 0:
#                     # Assuming all pets are enrolled, YOLO prediction of pet type could be wrong.
#                     house_sub_db = enrolled_hh_pets_db.copy()
#                 min_dist = 99999
#                 for i in range(len(house_sub_db)):
#                     row = house_sub_db.iloc[i]
#                     dist = np.sum(
#                         np.square(np.subtract(row["embedding"], pred_embed[rows]))
#                     )
#                     if dist < min_dist:
#                         min_dist = dist
#                         d[row["pet_id"]] = dist
#                 ids.append(sorted(d.items(), key=lambda x: x[1])[0][0])
#             pred_pet_id.append(ids)

#         else:
#             house_sub_db = enrolled_hh_pets_db[
#                 enrolled_hh_pets_db["pet_type"] == pred_pet_type
#             ]
#             if len(house_sub_db) == 0:
#                 # Assuming all pets are enrolled, YOLO prediction of pet type could be wrong.
#                 house_sub_db = enrolled_hh_pets_db.copy()
#             min_dist = 99999
#             pred_pet_id = None
#             for i in range(len(house_sub_db)):
#                 row = house_sub_db.iloc[i]
#                 dist = np.sum(np.square(np.subtract(row["embedding"], pred_embed)))
#                 if dist < min_dist:
#                     min_dist = dist
#                 pred_pet_id = row["pet_id"]

#         res.append(pred_pet_id)

#     return res
