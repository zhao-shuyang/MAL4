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

import time
import os

usegpu = True
device = torch.device("cuda" if usegpu else "cpu")
#torch.cuda.set_device(2)

target_input_length = 501
target_input_dim = 128

db_path = 'database/db_Audioset_UrbanSound.sqlite'
feature_path = '/wrk/shuyzhao/mel_combine_44100_128.hdf5'

tmp_model_weight_path = 'model/current_pair.pkl'
best_model_weight_path = 'model/best_pair.pkl'


class l18(nn.Module):
    def __init__(self):
        super(l18,self).__init__() 
        self.layer1 = nn.Sequential(nn.Conv2d(1,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer2 = nn.Sequential(nn.Conv2d(16,16,kernel_size=3,padding=1),nn.BatchNorm2d(16),nn.ReLU())
        self.layer3 = nn.MaxPool2d(2)

        self.layer4 = nn.Sequential(nn.Conv2d(16,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.layer5 = nn.Sequential(nn.Conv2d(32,32,kernel_size=3,padding=1),nn.BatchNorm2d(32),nn.ReLU())
        self.layer6 = nn.MaxPool2d(2)

        self.layer7 = nn.Sequential(nn.Conv2d(32,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.layer8 = nn.Sequential(nn.Conv2d(64,64,kernel_size=3,padding=1),nn.BatchNorm2d(64),nn.ReLU())
        self.layer9 = nn.MaxPool2d(2)

        self.layer10 = nn.Sequential(nn.Conv2d(64,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.layer11 = nn.Sequential(nn.Conv2d(128,128,kernel_size=3,padding=1),nn.BatchNorm2d(128),nn.ReLU())
        self.layer12 = nn.MaxPool2d(2)

        self.layer13 = nn.Sequential(nn.Conv2d(128,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.layer14 = nn.Sequential(nn.Conv2d(256,256,kernel_size=3,padding=1),nn.BatchNorm2d(256),nn.ReLU())
        self.layer15 = nn.MaxPool2d(2) #

        self.layer16 = nn.Sequential(nn.Conv2d(256,512,kernel_size=3,padding=1),nn.BatchNorm2d(512),nn.ReLU())
        self.layer17 = nn.MaxPool2d(2) # 
        self.layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2),nn.ReLU(), nn.Tanh())
        
    def forward(self,x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = self.layer14(out)
        out = self.layer15(out)
        out = self.layer16(out)
        out = self.layer17(out)
        out = self.layer18(out)
        
        #print (out.size())
        
        out = torch.transpose(out, 1, 2)
        out = torch.squeeze(out, 3)
        
        return out

    def load_weight(self, weight_path):
        state_dict = torch.load(weight_path, map_location=lambda storage, loc: storage)
        self.load_state_dict(state_dict)

def padding(f, trg_n=target_input_length):
    n,d = f.shape
    r = int(trg_n/n)
    m = trg_n%n
    output = np.zeros((trg_n, d))
    for i in range(r):
        output[n*i:n*(i+1)] = f
    output[r*n:] = f[:m]
    return output

def normalize(f):
    #h5r = h5py.File(stats_path, 'r') 
    n,d = f.shape
    g = (f-f_mean_mat)/f_std_mat

    return(g)

class BatchLoader():
    def __init__(self):
        self.db = datasetSQL.LabelSet(db_path)
        self.h5r = h5py.File(feature_path, 'r')

    def load(self, n_same=20, n_diff=30):
        self.db.cursor.execute("SELECT COUNT(*) FROM classes;")
        n_class = self.db.cursor.fetchone()[0]
        order = np.random.permutation(range(1, n_class+1))

        batch_data = np.zeros(((n_same + n_diff), 2, target_input_length, target_input_dim))
        batch_target = np.zeros((n_same + n_diff))

        i = 0
        i_hop = 0
        
        while i < n_same:
            sql = """
            SELECT segment_id FROM labels WHERE class_id = {0} ORDER BY RANDOM() LIMIT 2
            """.format(order[i+i_hop])
            self.db.cursor.execute(sql)
            records = self.db.cursor.fetchall()
            
            if (len(records)<2): #In case of removed classes
                i_hop += 1
                continue

            pair = records[0][0].decode('utf-8'), records[1][0].decode('utf-8')
            label_target = {}

            for j in range(2):
                f = self.h5r[pair[j]][:]
                if len(f) != target_input_length:
                    f = padding(f)
                #f = normalize(f)
                
                batch_data[i,j,:,:] = f

                
                sql = """
                SELECT class_id FROM labels WHERE segment_id = '{0}'
                """.format(pair[j])
                self.db.cursor.execute(sql)
                records = self.db.cursor.fetchall()
                label_target[j] = np.zeros(n_class)

                for record in records:
                    #print (record)
                    label_target[j][record[0]-1] = 1
            #print (np.sum(label_target[0]), np.sum(label_target[1]))         
            batch_target[i] = similarity.cosine_similarity(label_target[0], label_target[1])
            i += 1
            
        while i < n_same + n_diff:
            sql = """
            SELECT segment_id FROM labels WHERE class_id = {0} ORDER BY RANDOM() LIMIT 1
            """.format(order[i+i_hop])
            
            self.db.cursor.execute(sql)
            records = self.db.cursor.fetchall()
            #p = int(np.random.random() * len(records))
            if (len(records)<1): #In case of removed classes
                i_hop += 1
                continue

            pair = (records[0][0].decode('utf-8'), )

            sql = """
            SELECT segment_id FROM segments WHERE segment_id NOT IN
            (SELECT segment_id FROM labels WHERE class_id = {0} )
            ORDER BY RANDOM() LIMIT 1
            """.format(order[i+i_hop])
            self.db.cursor.execute(sql)
            records = self.db.cursor.fetchone()
            #q = int(np.random.random() * len(records))
            pair = pair + (records[0].decode('utf-8'), )

            label_target = {}

            for j in range(2):            
                f = self.h5r[pair[j]][:]
                if len(f) != target_input_length:
                    f = padding(f)
                #f = normalize(f)
                batch_data[i,j,:,:] = f
                
                sql = """
                SELECT class_id FROM labels WHERE segment_id = '{0}'
                """.format(pair[j])
                self.db.cursor.execute(sql)
                records = self.db.cursor.fetchall()
                label_target[j] = np.zeros(n_class)
                for record in records:
                    label_target[j][record[0]-1] = 1.0
            
            batch_target[i] = similarity.cosine_similarity(label_target[0], label_target[1])
            #batch_target = np.round(batch_target)
            i += 1
        return batch_data, batch_target


def train(db_path, feature_path):
    batch_loader = BatchLoader()
    netx = l18().to(device)
    if os.path.isfile(tmp_model_weight_path):
        netx.load_weight(tmp_model_weight_path)
    print(netx)
    optimizer = optim.Adam(netx.parameters(), lr=0.00005)
    #optimizer = optim.SGD(netx.parameters(), lr=0.00001)
    loss_function = nn.BCELoss()
    netx.train()

    best_num = 0
    running_loss = np.ones(100)
    for i_batch in range(100000):
        batch_data, target = batch_loader.load(30,40)
        torch_data =  torch.from_numpy(batch_data).float().to(device)
        torch_target = torch.from_numpy(target).float().to(device)

        
        for i in range(1):
            optimizer.zero_grad()
            emb0 = netx(torch_data[:,0,:,:].unsqueeze(1))
            #emb0B = netx(torch_data[:,0,32:,:].unsqueeze(1))
            
            emb1 = netx(torch_data[:,1,:,:].unsqueeze(1))
            #emb1B = netx(torch_data[:,1,32:,:].unsqueeze(1))

            #emb0 = torch.cat((emb0A, emb0B),1)
            #emb1 = torch.cat((emb1A, emb1B),1)
            
            emb0 = torch.max(emb0, 1)[0]
            emb1 = torch.max(emb1, 1)[0]
            pred = Fx.cosine_similarity(emb0, emb1)
            #print (emb0, emb1, pred, torch_target)
            loss = loss_function(pred, torch_target)
            if i == 0:
                running_loss = np.roll(running_loss, 1)
                running_loss[0] = loss
            #print (torch.sum(emb0>0,1))
            #print (emb0.size(1))
            #nz_rate = torch.mean(torch.max(torch.cat((torch.sum(emb0>0,1).unsqueeze(1), torch.sum(emb1>0,1).unsqueeze(1)),1).float()/emb0.size(1),1)[0]) #non_zero rate            
            
            #loss += nz_rate #Penalize non-zero output
            print (i_batch, loss, np.mean(running_loss))
            loss.backward()
            optimizer.step()

        if i_batch % 10 == 0:
            print (np.mean(running_loss))
        
        if i_batch % 1000 == 50:
            torch.save(netx.state_dict(), tmp_model_weight_path)            
            print ("Model has been saved...")

            criteria_i = evaluate(tmp_model_weight_path, 'database/db_esc10.sqlite', 'database/mel_esc10.hdf5')
            if criteria_i > best_num:
                copyfile(tmp_model_weight_path, best_model_weight_path)

def evaluate(weight_path, db_path, feature_path):
    db = datasetSQL.LabelSet(db_path)
    h5r = h5py.File(feature_path, 'r')
    h5w = h5py.File('database/esc_tmp_2.hdf5', 'w')    
    db.cursor.execute("SELECT segment_id FROM segments WHERE audio_file NOT NULL;")
    segment_list = [record[0].decode('utf-8') for record in db.cursor.fetchall()]
    n_segment = len(segment_list)

    netx = l18().to(device)
    netx.load_weight(weight_path)
    netx.eval()
    
    for k in h5r.keys():
        n,D = h5r[k][:].shape
        data = np.zeros((1,1,n,D))
        f = h5r[k][:]
        #f -= f_mean
        #f /= f_std
        data[0,0,:,:] = f 
        torch_data = torch.from_numpy(data).float().to(device)
        with torch.no_grad():
            embA = netx(torch_data)
            embB = netx(torch_data[:,:,64:,:])
            emb = torch.cat((embA,embB),1)
            emb = torch.max(emb, 1)[0]
            
        #print (k, emb)

        h5w.create_dataset(k, data=emb.cpu().numpy()[0])
    h5w.close()


    h5r2 = h5py.File('database/esc_tmp_2.hdf5', 'r')
    h5w2 = h5py.File('database/esc_tmp_dist_2.hdf5', 'w')
    similarity.Cosine(h5r2, h5w2)
    h5w2.close()
    
    h5r3 = h5py.File('database/esc_tmp_dist_2.hdf5', 'r')
    return (similarity_analysis.mAP(h5r3, db))
    
    
            
if __name__ == '__main__':    
    train(db_path, feature_path)
    #evaluate(tmp_model_weight_path, db_eval_path, feature_eval_path)
    evaluate(tmp_model_weight_path, 'database/db_esc10.sqlite', 'database/mel_esc10.hdf5')
