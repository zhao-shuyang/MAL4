#!/usr/bin/python
__author__ = "Zhao Shuyang"
__email__ = "shuyang.zhao@tut.fi"

import numpy as np
import h5py
import datasetSQL

def precK(dist_h5, label_db, K=39):
    f_obj = h5py.File(dist_h5,'r')
    db = datasetSQL.LabelSet(label_db)
    #dist_mat = f_obj["AudiosetT"][:]
    dist_mat = f_obj["dist_mat"][:]
    N, N = dist_mat.shape
    precs = np.zeros(N)
    for i in range(N):
        ranked_indices = np.argsort(dist_mat[i])
        sql = """
        SELECT class_name FROM labels ORDER BY audio_file ASC;
        """
        db.cursor.execute(sql)
        class_list = [class_name[0] for class_name in db.cursor.fetchall()]
        similar_first_class_memberships = [class_list[index] for index in ranked_indices]
        print (class_list[i])
        print (similar_first_class_memberships[:K])
        match_count = 0
        for k in range(K):
            if class_list[i] == similar_first_class_memberships[k]:
                match_count += 1
        precs[i] = float(match_count)/K
    print (np.max(precs), np.min(precs), np.mean(precs))


def mAP(f_obj, db):
    #f_obj = h5py.File(dist_h5,'r')
    #db = datasetSQL.LabelSet(label_db)
    dist_mat = f_obj["dist_mat"][:]
    #dist_mat = f_obj["MFCCs"][:]
    N, N = dist_mat.shape
    precs = np.zeros(N)
    sql = """
    SELECT segment_id, class_id FROM labels ORDER BY segment_id ASC;
    """
    db.cursor.execute(sql)
    records = db.cursor.fetchall()

    segment_list = [record[0] for record in records]
    class_list = [record[1] for record in records]
    #print (class_list)
    for i in range(N):
        #print (dist_mat[i])
        ranked_indices = np.argsort(dist_mat[i])
        #print (dist_mat[i][ranked_indices])
        #print (segment_list)
        #print (class_list)
        sql = """
        SELECT COUNT(*) FROM labels WHERE class_id = {0};
        """.format(class_list[i])
        #print (sql)
        db.cursor.execute(sql)
        K = db.cursor.fetchone()[0] - 1
        
        similar_first_class_memberships = [class_list[index] for index in ranked_indices]
        
        #print (similar_first_class_memberships[:K+1])
        match_count = 0
        score = 0
        for k in range(1,K+1):
            if class_list[i] == similar_first_class_memberships[k]:
                match_count += 1.
            score += match_count/(k)
        precs[i] = score/K
    print (np.max(precs), np.min(precs), np.mean(precs))
    return (np.mean(precs))

def mAP2(f_obj, db):
    #f_obj = h5py.File(dist_h5,'r')
    #db = datasetSQL.LabelSet(label_db)
    dist_mat = f_obj["dist_mat"][:]
    #dist_mat = f_obj["MFCCs"][:]
    N, N = dist_mat.shape
    precs = np.zeros(N)
    sql = """
    SELECT segment_id, class_id FROM labels ORDER BY segment_id ASC;
    """
    db.cursor.execute(sql)
    records = db.cursor.fetchall()

    segment_list = [record[0] for record in records]

    class_list = [record[1] for record in records]
    
    mAP = 0
    for i in range(N):
        ranked_indices = np.argsort(dist_mat[i])
        sql = """
        SELECT COUNT(*) FROM labels WHERE class_id = '{0}';
        """.format(class_list[i])
        #print (sql)
        db.cursor.execute(sql)
        K = db.cursor.fetchone()[0]

        correct_found = 0
        aveP = 0
        last_recall = 0
        recall_previous = 0

        #print (class_list, ranked_indices)
        similar_first_class_memberships = [class_list[index] for index in ranked_indices]
        
        for j in range(N):
            if similar_first_class_memberships[j] == similar_first_class_memberships[0]:
                correct_found += 1
                #recall_previous = recall
            
            precision = correct_found / (j + 1)
            recall = correct_found / K
                
            aveP += precision * (np.abs(recall_previous - recall))
            recall_previous = recall
            
            #print (correct_found, j, aveP)

        mAP += aveP/N
    print (mAP)
    return(mAP)
        
#def alpha_neighborhood()
if __name__ == '__main__':
    import sys
    print (mAP2(sys.argv[1], sys.argv[2]))
    #print (precK(sys.argv[1], sys.argv[2]))
