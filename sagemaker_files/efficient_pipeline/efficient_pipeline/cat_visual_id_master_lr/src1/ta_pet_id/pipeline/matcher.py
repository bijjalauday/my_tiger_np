# """
# This module contains core utilities used for matching embeddings and getting predicted pet IDs
# """

import numpy as np
from sklearn.linear_model import LogisticRegression
import pandas as pd
"""
This module contains core utilities used for matching embeddings and getting predicted pet IDs
"""

import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import LabelEncoder


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
    res1=[]
    res_orig=[]
    l=[]
    
    lr=LogisticRegression()
    # lr1=KNeighborsClassifier(n_neighbors=1)
    for i in enrolled_hh_pets_db["embedding"].values:
        l.append(i)
    df=pd.DataFrame(l)
    df["pet_id"]=enrolled_hh_pets_db["pet_id"].values
    le=LabelEncoder()
    df["orig"]=le.fit_transform(df["pet_id"])
    if enrolled_hh_pets_db["pet_id"].nunique()>1:
        ch=0
        lr.fit(df.iloc[:,:64],df["orig"])
        # lr1.fit(df.iloc[:,:64],df["orig"])
        # print("lr",end=" ")
    else:
        print("lr",end=" ")
        ch=1
    
    


    for pred_embed, pred_pet_type in zip(pred_pet_embds, pred_pet_types):
        if ch==1:
            print("lr",end=" ")
            res1.append(1)
            res.append(le.inverse_transform([0]))
            res_orig.append(le.inverse_transform([0]))
            n_classes=1
        
        else:
            if pred_embed is None or pred_pet_type is None:
                res.append(None)
                continue

            # house_sub_db = enrolled_hh_pets_db[enrolled_hh_pets_db['pet_type'] == pred_pet_type]
            # if len(house_sub_db) == 0:
            #     # Assuming all pets are enrolled, YOLO prediction of pet type could be wrong.
            #     house_sub_db = enrolled_hh_pets_db.copy()
            house_sub_db = enrolled_hh_pets_db.copy()
            # min_dist = -2
            # pred_pet_id = None
            # for i in range(len(house_sub_db)):
            #     row = house_sub_db.iloc[i]
            #     dist = cosine_similarity([row['embedding']], [pred_embed])[0][0]
            #     if dist > min_dist:
            #         min_dist = dist
            #         pred_pet_id = row['pet_id']
            pred_embed=list(pred_embed)
            n_classes=enrolled_hh_pets_db["pet_id"].nunique()
            thresh=(100/n_classes)+(100/n_classes)*((n_classes+1)/10)
            thresh=thresh/100
            # try:
            pred_pet_id=lr.predict([pred_embed])[0]
            proba=list(lr.predict_proba([pred_embed])[0])
            # except:
            #     pred_pet_id=lr1.predict([pred_embed])[0]
            #     proba=list(lr1.predict_proba([pred_embed])[0])
            # print(proba,thresh,pred_pet_id)
            res1.append(proba[pred_pet_id])
            res_orig.append(le.inverse_transform([pred_pet_id])[0])
            if proba[pred_pet_id]<thresh:
                pred_pet_id=-1
                res.append(pred_pet_id)
            else:
                res.append(le.inverse_transform([pred_pet_id])[0])
    print(len(res),len(res1),n_classes)
    return res,res1,res_orig


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
