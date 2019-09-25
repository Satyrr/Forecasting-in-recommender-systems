import numpy as np
import pandas as pd

def NDPM(predictions):

    true_ratings_by_user_id = {}
    est_ratings_by_user_id = {}
    for p in predictions:

        if p.uid not in true_ratings_by_user_id:
            true_ratings_by_user_id[p.uid] = []
            est_ratings_by_user_id[p.uid] = []

        true_ratings_by_user_id[p.uid].append(p.r_ui)
        est_ratings_by_user_id[p.uid].append(p.est)
    
    NDPMs = []
    for user_id in [u_id for u_id in true_ratings_by_user_id if len(true_ratings_by_user_id[u_id]) > 1]:
        
        if np.all(np.array(true_ratings_by_user_id[user_id]) == true_ratings_by_user_id[user_id][0]):
            continue

        ndpm = NDMP_for_user(true_ratings_by_user_id[user_id], est_ratings_by_user_id[user_id])
        NDPMs.append(ndpm)

    return np.array(NDPMs).mean()
def NDMP_for_user(true_ratings, estimated_ratings):
    true_ratings, estimated_ratings = np.array(true_ratings), np.array(estimated_ratings)
    true_ratings_differences = true_ratings[:, None] - true_ratings
    true_ratings_differences[true_ratings_differences < 0 ] = -1
    true_ratings_differences[true_ratings_differences == 0 ] = 0
    true_ratings_differences[true_ratings_differences > 0 ] = 1

    estimated_ratings_differences = estimated_ratings[:, None] - estimated_ratings
    estimated_ratings_differences[estimated_ratings_differences < 0 ] = -1
    estimated_ratings_differences[estimated_ratings_differences == 0 ] = 0
    estimated_ratings_differences[estimated_ratings_differences > 0 ] = 1

    order_asserts_matrix = np.triu(true_ratings_differences * estimated_ratings_differences)
    
    Cplus = order_asserts_matrix[order_asserts_matrix == 1].sum()
    Cminus = np.abs(order_asserts_matrix[order_asserts_matrix == -1].sum())

    Cu = np.triu(true_ratings_differences * true_ratings_differences).sum()
    
    Cu0 = Cu - (Cplus + Cminus)

    return (Cminus + 0.5*Cu0)/Cu