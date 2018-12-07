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

usegpu = False
device = torch.device("cuda" if usegpu else "cpu")
training_batch_size = 64
target_input_length = 626
target_input_dim = 128

featType = 'layer18'
db_path = 'database/db_train.sqlite'
feature_path = 'database/mel_train.hdf5'
db_eval_path = 'database/db_eval.sqlite'
feature_eval_path = 'database/mel_eval.hdf5'
tmp_model_weight_path = 'model/current_tmp.pkl'
best_model_weight_path = 'model/best_weight.pkl'


class weak_mxh64_1024(nn.Module):
    def __init__(self,nclass,glplfn=Fx.max_pool2d):
        super(weak_mxh64_1024,self).__init__() 
        self.globalpool = glplfn
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
        
        self.layer18 = nn.Sequential(nn.Conv2d(512,1024,kernel_size=2),nn.BatchNorm2d(1024),nn.ReLU())
        self.layer19 = nn.Sequential(nn.Conv2d(1024,nclass,kernel_size=1),nn.Sigmoid())

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
        out1 = self.layer19(out)
        out = self.globalpool(out1,kernel_size=out1.size()[2:])
        out = out.view(out.size(0),-1)
        return out

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
    
    db.cursor.execute("SELECT segment_id FROM segments WHERE audio_file NOT NULL LIMIT 640;")
    segment_list = [record[0].decode('utf-8') for record in db.cursor.fetchall()]
    n_segment = len(segment_list)

    netx = weak_mxh64_1024(n_class)
    optimizer = optim.Adam(netx.parameters())
    loss_function = nn.BCELoss()
    netx.train()
    best_epoch = 0
    for i_epoch in range(20):
        losses = 0
        order_list = np.random.permutation(range(n_segment))
        for start_index in range(0, n_segment, training_batch_size):
            if start_index + training_batch_size > n_segment:
                    continue
            print (start_index,n_segment)
            batch_data = np.zeros((training_batch_size, 1, target_input_length, target_input_dim))
            batch_target = np.zeros((training_batch_size, n_class))
            for i in range(training_batch_size):

                segment_id = segment_list[order_list[start_index + i]]
                f = h5r[segment_id][:]
                if len(f) != target_input_length:
                    f = padding(f)
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
            torch_output = netx(torch_train)
            #torch_output = torch.round(torch_output)
            loss = loss_function(torch_output, torch_target)
            losses += loss
            print ("Batch training loss {0}".format(loss))
            loss.backward()
            optimizer.step()
            
        print ("epoch loss {0}".format(losses))
        torch.save(netx.state_dict(), 'model/current_tmp.pkl')            
        print ("Model has been saved...")
        criteria_i = evaluate(tmp_model_weight_path, db_eval_path, feature_eval_path)
        if criteria_i > best_epoch:
            copyfile(tmp_model_weigtt_path, best_model_weight_path)

def evaluate(weight_path, db_path, feature_path):
    db = datasetSQL.LabelSet(db_path)
    h5r = h5py.File(feature_path, 'r')    
    db.cursor.execute("SELECT COUNT(*) FROM classes;")
    n_class = db.cursor.fetchone()[0]
    db.cursor.execute("SELECT segment_id FROM segments WHERE audio_file NOT NULL LIMIT 320;")
    segment_list = [record[0].decode('utf-8') for record in db.cursor.fetchall()]
    n_segment = len(segment_list)

    netx = weak_mxh64_1024(n_class)
    netx.load_state_dict(torch.load(weight_path))
    netx.eval()

    batch_data = np.zeros((training_batch_size, 1, target_input_length, target_input_dim))
    batch_target = np.zeros((training_batch_size, n_class))
    loss_function = nn.BCELoss()
    loss = 0
    total_tp = 0
    total_union = 0
    for start_index in range(0, n_segment, training_batch_size):
        print (start_index,n_segment)
        if start_index + training_batch_size > n_segment:
            continue
        batch_data = np.zeros((training_batch_size, 1, target_input_length, target_input_dim))
        batch_target = np.zeros((training_batch_size, n_class))
        for i in range(training_batch_size):
            segment_id = segment_list[start_index + i]
            f = h5r[segment_id][:]
            if len(f) != target_input_length:
                f = padding(f)
            batch_data[i,0,:,:] = f

            sql = """
            SELECT class_id FROM labels WHERE segment_id = '{0}'
            """.format(segment_id)

            db.cursor.execute(sql)
            records = db.cursor.fetchall()
            for record in records:
                batch_target[i, record[0]-1] = 1
                
        torch_data = torch.from_numpy(batch_data).float().to(device)
        torch_target =  torch.from_numpy(batch_target).float()
        with torch.no_grad():
            torch_output = netx(torch_data)
        #print (torch_output)
        predictions = torch.round(torch_output).cpu()
        union = predictions + torch_target
        union[union > 0] = 1
        batch_tp = torch.sum(predictions * torch_target)
        batch_total = torch.sum(union)
        batch_jaccard = batch_tp/batch_total
        print (batch_tp, batch_total, batch_jaccard)
        loss += loss_function(torch_output, torch_target)
        total_tp += batch_tp
        total_union += batch_total
    jaccard = total_tp/total_union
    print (jaccard, loss)
    return (jaccard)
        
if __name__ == '__main__':    
    train(db_path, feature_path)
    evaluate(tmp_model_weight_path, db_eval_path, feature_eval_path)
