import torch
import torch.nn as nn
import torch.nn.functional as Fx
import torch.nn.init as init
from torch import optim

import numpy as np
import h5py
from collections import OrderedDict
from operator import itemgetter
from shutil import copyfile

import datasetSQL
import similarity
import similarity_analysis
import network_arch

import os

usegpu = True
device = torch.device("cuda" if usegpu else "cpu")
#torch.cuda.set_device(2)

training_batch_size = 128
target_input_length = 501
target_input_dim = 128
embedding_length = 1024

db_path = 'database/db_Audioset_UrbanSound.sqlite'
feature_path = 'database/mel_combine_44100_128.hdf5'
stats_path = 'database/mel_44100_stats.hdf5'

#stats_path = 'database/mel_44100_stats.hdf5'
#h5r = h5py.File(stats_path, 'r') 
#f_mean, f_std = h5r['mean'][:], h5r['std'][:]    
#f_mean_mat = np.repeat([f_mean], target_input_length, axis=0)
#f_std_mat = np.repeat([f_std], target_input_length, axis=0)

#db_eval_path = 'database/db_eval.sqlite'
#feature_eval_path = '/database/mel_eval_44100_128.hdf5'
tmp_model_weight_path0 = 'model/current_unit_Urban.pkl'
best_model_weight_path0 = 'model/best_unit_Urban.pkl'
tmp_model_weight_path1 = 'model/current_unit_cls_Urban.pkl'
best_model_weight_path1 = 'model/best_unit_cls_Urban.pkl'


def padding(f, trg_n=target_input_length):
    n,d = f.shape
    r = int(trg_n/n)
    m = trg_n%n
    output = np.zeros((trg_n, d))
    for i in range(r):
        output[n*i:n*(i+1)] = f
    output[r*n:] = f[:m]
    return output


def train(db_path, feature_path):
    db = datasetSQL.LabelSet(db_path)
    h5r = h5py.File(feature_path, 'r')
    db.cursor.execute("SELECT COUNT(*) FROM classes;")
    n_class = db.cursor.fetchone()[0]

    cascade_net = network_arch.CascadeNet(n_class).to(device)
    
    if os.path.isfile(tmp_model_weight_path1):
        cascade_net.load_weight(tmp_model_weight_path1)
    
    #params = list(emb_net.parameters()) + list(cls_net.parameters())    
    optimizer = optim.Adam(cascade_net.parameters(), lr=0.0001)
    #optimizer = optim.Adam([{'params': cascade_net.emb_net.parameters(), 'lr':1e-4},{'params': cascade_net.cls_net.parameters(), 'lr': 1e-5}])
    
    loss_function = nn.BCELoss()

    cascade_net.train()
    best_epoch = 0
    class_list = np.random.permutation(range(2,n_class+1))

    for i_epoch in range(3000):
        losses = 0
        #k_class = 30 + int(i_epoch/1)
        db.cursor.execute("SELECT segment_id FROM segments")
    
        #sql = """
        #SELECT DISTINCT(labels.segment_id) FROM labels INNER JOIN segments
        #ON labels.segment_id = segments.segment_id
        #WHERE segments.audio_file NOT NULL
        #"""
        #labels.class_id IN ({0})
        #""".format(','.join([str(cn) for cn in class_list[:k_class]]))
        #print (sql)
        #db.cursor.execute(sql)
        segment_list = [record[0].decode('utf-8') for record in db.cursor.fetchall()]
        n_segment = len(segment_list)
        print (i_epoch, n_segment)

        order_list = np.random.permutation(range(n_segment))
        for start_index in range(0, n_segment, training_batch_size):
            if start_index + training_batch_size > n_segment:
                    continue
            batch_data = np.zeros((training_batch_size, 1, target_input_length, target_input_dim))
            batch_target = np.zeros((training_batch_size, n_class))
            optimizer.zero_grad()
            
            for i in range(training_batch_size):
                segment_id = segment_list[order_list[start_index + i]]
                #print (start_index+i, order_list[start_index + i], segment_id)
                f = h5r[segment_id][:]
                if len(f) != target_input_length:
                    f = padding(f)
                #f = normalize(f)
                batch_data[i,0,:,:] = f

                sql = """
                SELECT class_id FROM labels WHERE segment_id = '{0}'
                """.format(segment_id)
                db.cursor.execute(sql)
                records = db.cursor.fetchall()
                for record in records:
                    batch_target[i, record[0]-1] = 1
                
            torch_train = torch.from_numpy(batch_data).float().to(device)
            torch_target =  torch.from_numpy(batch_target).float().to(device)
            for i in range(1):
                torch_output = cascade_net(torch_train)
                #print (torch_output)
                loss = loss_function(torch_output, torch_target)
                loss.backward()
                optimizer.step()
                print (start_index, loss)
                losses += loss
            """
            predictions = torch.round(torch_output)
            union = predictions + torch_target
            union[union > 0] = 1
            batch_total = torch.sum(union)
            batch_tp = torch.sum(predictions * torch_target)
            print (batch_tp, batch_total, batch_tp/batch_total)
            """
        print ("epoch {0} loss: {1}".format(i_epoch, losses))
        torch.save(cascade_net.state_dict(), tmp_model_weight_path1)
        torch.save(cascade_net.emb_net.state_dict(), tmp_model_weight_path0)            
        
        print ("Model has been saved...")

        if i_epoch %1 == 0: 
            criteria_i = evaluate(tmp_model_weight_path0, 'database/db_esc50.sqlite', 'database/mel_esc50.hdf5')
            if criteria_i > best_epoch:
                best_epoch = criteria_i
                copyfile(tmp_model_weight_path0, best_model_weight_path0)
                copyfile(tmp_model_weight_path1, best_model_weight_path1)

def evaluate(weight_path, db_path, feature_path):
    db = datasetSQL.LabelSet(db_path)
    h5r = h5py.File(feature_path, 'r')
    h5w = h5py.File('/tmp/esc_tmp_1.hdf5', 'w')    
    n_class = 527 #quick hack now, this depends on the source problem
    db.cursor.execute("SELECT segment_id FROM segments WHERE audio_file NOT NULL ORDER BY segment_id ASC;")
    segment_list = [record[0].decode('utf-8') for record in db.cursor.fetchall()]
    n_segment = len(segment_list)
    h5w.create_dataset('max', data=np.zeros((n_segment, embedding_length)))
    
    emb_net = network_arch.EmbNet().to(device)
    emb_net.load_weight(weight_path)
    emb_net.eval()
    
    for i, segment_id in enumerate(segment_list):
        f = h5r[segment_id][:]

        if len(f) < 251:
            f = padding(f,251)
        n,D = f.shape
        data = np.zeros((1,1,n,D))        
        data[0,0,:,:] = f 
        torch_data = torch.from_numpy(data).float().to(device)
        
        with torch.no_grad():
            #pred = netx(torch_data)[0]
            """
            pred = emb_net(torch_data)
            #print (pred.size())

            if len(pred.size()) > 2:
                embedding = torch.max(pred, 2)[0]
                embedding = embedding.view(embedding.size(0),-1)
            else:
                embedding = pred
            """
            embedding = emb_net(torch_data)
            
        #print (embedding)
        h5w['max'][i] = embedding.cpu().numpy()[0]
        h5w.create_dataset(segment_id, data=embedding.cpu().numpy()[0])
    h5w.close()

    h5r2 = h5py.File('/tmp/esc_tmp_1.hdf5', 'r')
    h5w2 = h5py.File('/tmp/esc_tmp_dist_1.hdf5', 'w')
    similarity.Dist_gpu(h5r2, h5w2)
    h5w2.close()
    
    h5r3 = h5py.File('/tmp/esc_tmp_dist_1.hdf5', 'r')
    return (similarity_analysis.mAP2(h5r3, db))

    
    
            
if __name__ == '__main__':    
    train(db_path, feature_path)
    #evaluate('model/current_unit_Urban.pkl', 'database/db_esc50.sqlite', 'database/mel_esc50.hdf5')

    
