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
    res = []
    l=[]
    lr=LogisticRegression()
    for i in enrolled_hh_pets_db["embedding"].values:
        l.append(i)
    df=pd.DataFrame(l)
    df["pet_id"]=enrolled_hh_pets_db["pet_id"]
    le=LabelEncoder()
    df["orig"]=le.fit_transform(df["pet_id"])
    return df

