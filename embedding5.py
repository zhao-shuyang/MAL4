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

import time
import os

usegpu = True
device = torch.device("cuda" if usegpu else "cpu")
#torch.cuda.set_device(3)

target_input_length = 501
target_input_dim = 128
training_batch_size = 40
n_class = 527
embedding_length = 1024

db_path = 'database/db_Audioset_UrbanSound.sqlite'
feature_path = 'database/mel_combine_44100_128.hdf5'
stats_path = 'database/mel_44100_stats.hdf5'


tmp_model_weight_path0 = 'model/current_13_Urban.pkl'
best_model_weight_path0 = 'model/best_13_Urban.pkl'
tmp_model_weight_path1 = 'model/current_13_cls_Urban.pkl'
best_model_weight_path1 = 'model/best_13_cls_Urban.pkl'


def padding(f, trg_n=target_input_length):
    n,d = f.shape
    r = int(trg_n/n)
    m = trg_n%n
    output = np.zeros((trg_n, d))
    for i in range(r):
        output[n*i:n*(i+1)] = f
    output[r*n:] = f[:m]
    return output

def triplet_loss(output, target, alpha=0.2): #For cosine similarity
    trg_sim_p = Fx.cosine_similarity(target[:,0,:], target[:,1,:])
    trg_sim_n = Fx.cosine_similarity(target[:,0,:], target[:,2,:])
    trg_sim_r = Fx.cosine_similarity(target[:,1,:], target[:,2,:])
    
    sim_p = Fx.cosine_similarity(output[:,0,:], output[:,1,:])
    sim_n = Fx.cosine_similarity(output[:,0,:], output[:,2,:])
    sim_r = Fx.cosine_similarity(output[:,1,:], output[:,2,:])

    l = (sim_n - sim_p)/(1-sim_r) + alpha
    #l = sim_n/sim_p - 1 
    #l = sim_n - sim_p + (trg_sim_p - trg_sim_n) * alpha
    rect = nn.ReLU()
    l = rect(l)
    #l2 = sim_r*(1 - trg_sim_r)*alpha**2
    #print (l, sim_r)
    print (torch.nonzero(l).size(0))
    #l = l1 + l2
    #L[trg_sim_p < trg_sim_n] = 0 
    L = torch.mean(l)
    return L

class BatchLoader():
    def __init__(self):
        self.db = datasetSQL.LabelSet(db_path)
        self.h5r = h5py.File(feature_path, 'r')
        #self.cm = cluster.ClusterManager()
        #self.cm.K = 1
        
    def pick_anchors(self, model_path):
        print ("Picking anchors...")
        
        emb_net = network_arch.EmbNet().to(device)
        emb_net.load_weight(model_path)
        emb_net.eval()
        
        sql = """
        SELECT class_id FROM classes WHERE class_id IN
        (SELECT class_id FROM labels GROUP BY class_id
        HAVING COUNT (segment_id) > 1)
        AND leaf_node = 1
        """
        self.db.cursor.execute(sql)
        self.anchor_class_list = [record[0] for record in self.db.cursor.fetchall()]
        #print (len(self.anchor_class_list))

        
        self.anchor = []
        for class_i in self.anchor_class_list:
            #print (class_i)
            sql = """
            SELECT segment_id FROM segments WHERE segment_id IN
            (SELECT segment_id FROM labels WHERE class_id = {0})
            """.format(class_i)
            self.db.cursor.execute(sql)
            class_instances = [record[0].decode('utf-8') for record in self.db.cursor.fetchall()]
            #print (class_instances)
            n = len(class_instances)
            #class_tar_sim_mat = torch.zeros((n, n), dtype=torch.float).to(device)
            class_label_mat = torch.zeros((n, n_class), dtype=torch.float).to(device)
            
            for i,segment_id in enumerate(class_instances):
                sql = """
                SELECT class_id FROM labels WHERE segment_id = '{0}'
                """.format(segment_id)
                self.db.cursor.execute(sql)
                for record in self.db.cursor.fetchall():
                    class_label_mat[i, record[0]-1] = 1

            label_count = torch.sum(class_label_mat, dim=1)
            core_instances_index =  (label_count == torch.min(label_count)).nonzero()
            core_instances = [class_instances[segment_i] for segment_i in core_instances_index]

            
            
            n = len(core_instances)                   
            
            class_sim_mat = torch.zeros((n, n), dtype=torch.float).to(device)
            class_feat_mat = torch.zeros((n, embedding_length), dtype=torch.float).to(device)
            
            for i, segment_id in enumerate(core_instances):
                f = self.h5r[segment_id][:]
                if len(f) != target_input_length:
                    f = padding(f)
                
                f = torch.from_numpy(f).float().unsqueeze(0).unsqueeze(0).to(device)
                with torch.no_grad():
                    pred = emb_net(f)
            
                if len(pred.size()) > 2:
                    embedding = torch.max(pred, 2)[0]
                    embedding = embedding.view(embedding.size(0),-1)
                else:
                    embedding = pred
                #print (embedding)
                class_feat_mat[i,:] = embedding
                
            class_sim_mat = torch.mm(class_feat_mat, class_feat_mat.transpose(0,1))/torch.ger(torch.norm(class_feat_mat,dim=1), torch.norm(class_feat_mat,dim=1))
            #print (torch.sum(class_sim_mat, dim=1))
            medoid = torch.argmax(torch.sum(class_sim_mat, dim=1)).item()
            #print (medoid)
            self.anchor.append(class_instances[medoid])
        self.anchor = list(set(self.anchor))
        self.anchor.sort()
        print (len(self.anchor))
            
        sql = """
        SELECT segment_id from segments ORDER BY segment_id ASC
        """
        self.db.cursor.execute(sql)
        segment_list = [record[0].decode('utf-8') for record in self.db.cursor.fetchall()]
        anchor_mat = torch.zeros((len(self.anchor), embedding_length), dtype=torch.float).to(device)
        full_mat = torch.zeros((len(segment_list), embedding_length), dtype=torch.float).to(device)
        
        j = 0
        for i, segment_id in enumerate(segment_list):
            sql = """
            SELECT class_id FROM labels WHERE segment_id = '{0}'
            """.format(segment_id)
            self.db.cursor.execute(sql)
            for record in self.db.cursor.fetchall():
                full_mat[i, record[0]-1] = 1

            if segment_id in self.anchor:
                anchor_mat[j, :] = full_mat[i,:]
                j += 1
            
        self.anchor_label_mat = anchor_mat
        self.trg_sim_mat = torch.mm(anchor_mat, full_mat.transpose(0,1))/torch.ger(torch.norm(anchor_mat,dim=1), torch.norm(full_mat,dim=1))
        self.trg_sim_mat = self.trg_sim_mat.cpu().numpy()        
        #print (self.trg_sim_mat)

    def mine_triplet(self, model_path):
        print ("Constructing loss triplets...")
        emb_net = network_arch.EmbNet().to(device)
        emb_net.load_weight(model_path)
        emb_net.eval()
        
        sql = """
        SELECT segment_id from segments ORDER BY segment_id ASC
        """
        self.db.cursor.execute(sql)
        segment_list = [record[0].decode('utf-8') for record in self.db.cursor.fetchall()]
        self.triplets = []

        for segment_i in range(self.trg_sim_mat.shape[1]):
            a1_index = np.argmax(self.trg_sim_mat[:,segment_i])
            #a2_index = np.argmin(arr[:,segment_i])
            a2_index = np.random.randint(len(self.anchor))
            while a2_index == a1_index:
                a2_index = np.random.randint(len(self.anchor))

            self.triplets.append((segment_list[segment_i], self.anchor[a1_index], self.anchor[a2_index]))


    def mine_batch_triplet(self, model_path, batch_data, batch_target):
        print ("Mining highest loss triplets...")
        emb_net = network_arch.EmbNet().to(device)
        emb_net.load_weight(model_path)
        emb_net.eval()

        order = np.random.permutation(training_batch_size)
        selected_anchor = [self.anchor[index] for index in order[:training_batch_size]]
        anchor_data = batch_data
        anchor_target = self.anchor_label_mat[order]
        
        for i, segment_id in enumerate(selected_anchor):
            f = self.h5r[segment_id][:]
            if len(f) != target_input_length:
                f = padding(f)
            anchor_data[i,0,:,:] = f
            
            anchor_data = anchor_data.to(device)
            with torch.no_grad():
                anchor_output = emb_net(anchor_data)
                anchor_output = anchor_output.view(embedding.size(0),-1)

        for segment_i in range(self.trg_sim_mat.shape[1]):
            a1_index = np.argmax(self.trg_sim_mat[:,segment_i])
        

            
    def load_triplet(self, n_triplets=40):
        """
        Random triplets
        """
        self.db.cursor.execute("SELECT COUNT(*) FROM classes;")
        n_class = self.db.cursor.fetchone()[0]
        order = np.random.permutation(len(self.anchor))


        batch_data = torch.zeros((n_triplets, 3, target_input_length, target_input_dim))
        label_target = torch.zeros((n_triplets, 3, n_class))
            
        #batch_target_margin = np.zeros((n_triplets))

        i = 0

        while i < n_triplets:
            anchor = self.anchor[order[i]]
            anchor_class = self.anchor_class_list[order[i]]

            sql = """
            SELECT segment_id FROM segments

            WHERE segment_id IN
            (SELECT segment_id FROM labels WHERE class_id = {0} AND segment_id != '{1}')
            ORDER BY RANDOM() LIMIT 1
            """.format(anchor_class, anchor)
            self.db.cursor.execute(sql)
            pos = self.db.cursor.fetchone()[0].decode('utf-8')
            
            sql = """
            SELECT segment_id FROM segments
            WHERE segment_id NOT IN
            (SELECT segment_id FROM labels WHERE class_id = {0})
            ORDER BY RANDOM() LIMIT 1
            """.format(anchor_class)
            self.db.cursor.execute(sql)
            t2 = time.time()
            records = self.db.cursor.fetchall()
            neg = records[0][0].decode('utf-8')
            
            triplet = (anchor, pos, neg)
            for j in range(3):
                f = self.h5r[triplet[j]][:]
                if len(f) != target_input_length:
                    f = padding(f)
                
                batch_data[i,j,:,:] = torch.from_numpy(f)
                
                sql = """
                SELECT class_id FROM labels WHERE segment_id = '{0}'
                """.format(triplet[j])
                self.db.cursor.execute(sql)
                records = self.db.cursor.fetchall()

                for record in records:
                    label_target[i, j, record[0]-1] = 1
            i += 1 

        return batch_data, label_target
    
def train(db_path, feature_path):
    batch_loader = BatchLoader()
    emb_net = network_arch.EmbNet().to(device)
    cls_net = network_arch.ASclassifier(n_class).to(device)

    if os.path.isfile(tmp_model_weight_path0):
        emb_net.load_weight(tmp_model_weight_path0)
    else:
        torch.save(emb_net.state_dict(), tmp_model_weight_path0)

    if os.path.isfile(tmp_model_weight_path1):
        cls_net.load_weight(tmp_model_weight_path1)
    else:
        torch.save(cls_net.state_dict(), tmp_model_weight_path1)

    optimizer = optim.Adam(list(emb_net.parameters()) + list(cls_net.parameters()), lr=1e-4)
    
    #optimizer = optim.Adam([{'params': emb_net.parameters(), 'lr':1e-5},{'params': cls_net.parameters(), 'lr': 1e-4}])
    #optimizer0 = optim.Adam(list(emb_net.parameters()) + list(cls_net.parameters()), lr=1e-4)
    #optimizer1 = optim.Adam(emb_net.parameters(), lr=1e-6)
                           
    
    loss_f1 = nn.BCELoss()
    emb_net.train()
    cls_net.train()
    
    best_num = 0
    best_epoch = 0
    for i_epoch in range(1000):
        batch_loader.pick_anchors(tmp_model_weight_path0)
        batch_loader.mine_triplet(tmp_model_weight_path0)
        losses = 0
        order = np.random.permutation(len(batch_loader.triplets))
        n_triplets = len(batch_loader.triplets)
        
        for i_batch, start_index in enumerate(range(0, n_triplets, training_batch_size)):
            if start_index + training_batch_size < n_triplets:
                batch_size = training_batch_size
            else:
                batch_size = n_triplets - start_index

            torch_data = torch.zeros((batch_size, 3, target_input_length, target_input_dim), dtype=torch.float32)
            torch_target = torch.zeros((batch_size, 3, n_class), dtype=torch.float)
        
            for i_input in range(batch_size):
                #ann, pos, neg = batch_loader.triplets[start_index + i_input]
                triplet = batch_loader.triplets[order[start_index + i_input]]
                #print (triplet)
                for j in range(3):
                    f = batch_loader.h5r[triplet[j]][:]
                    if len(f) != target_input_length:
                        f = padding(f)
                    torch_data[i_input,j,:,:] = torch.from_numpy(f).float()
                    
                    sql = """
                    SELECT class_id FROM labels WHERE segment_id = '{0}'
                    """.format(triplet[j])
                    batch_loader.db.cursor.execute(sql)
                    records = batch_loader.db.cursor.fetchall()

                    for record in records:
                        torch_target[i_input, j, record[0]-1] = 1

            torch_data = torch_data.to(device)
            torch_target = torch_target.to(device)
        
            for i in range(1):
                optimizer.zero_grad()
                #optimizer1.zero_grad()                
			
                emb_output = torch.zeros(torch_data.size(0), 3, 1024).float().to(device)
                
                emb_output[:,0,:] = emb_net(torch_data[:,0,:,:].unsqueeze(1))
                emb_output[:,1,:] = emb_net(torch_data[:,1,:,:].unsqueeze(1))
                emb_output[:,2,:] = emb_net(torch_data[:,2,:,:].unsqueeze(1))

                cls_output = cls_net(emb_output[:,0,:])
                cls_target = torch_target[:,0,:]

                loss1 = loss_f1(cls_output, cls_target)
                loss2 = triplet_loss(emb_output, torch_target)
                #loss3 = torch.mean(torch.norm(emb_output[:,0,:], dim=1))
                if i_batch %2 == 0:
                    loss = loss1
                else:
                    loss = 0.01*loss2
                losses += loss
                print (i_batch, loss1.item(), loss2.item())

                #loss1.backward(retain_graph=True)
                loss.backward()
                optimizer.step()
                #optimizer1.step()

                
        print ("epoch {0} loss: {1}".format(i_epoch, losses))
        #torch.save(netx.state_dict(), tmp_model_weight_path1)
        torch.save(emb_net.state_dict(), tmp_model_weight_path0)            
        torch.save(cls_net.state_dict(), tmp_model_weight_path1)            
        
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
    h5w = h5py.File('/tmp/esc_tmp_5.hdf5', 'w')    
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

        if len(f) < 201:
            f = padding(f,201)
        n,D = f.shape
        data = np.zeros((1,1,n,D))        
        data[0,0,:,:] = f 
        torch_data = torch.from_numpy(data).float().to(device)
        with torch.no_grad():
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
        #print (k, embedding)
        h5w['max'][i] = embedding.cpu().numpy()[0]
        h5w.create_dataset(segment_id, data=embedding.cpu().numpy()[0])
    h5w.close()

    h5r2 = h5py.File('/tmp/esc_tmp_5.hdf5', 'r')
    h5w2 = h5py.File('/tmp/esc_tmp_dist_5.hdf5', 'w')
    similarity.Dist_gpu(h5r2, h5w2)
    h5w2.close()
    
    h5r3 = h5py.File('/tmp/esc_tmp_dist_5.hdf5', 'r')
    return (similarity_analysis.mAP2(h5r3, db))


    
    
    
            
if __name__ == '__main__':    
    train(db_path, feature_path)
    #evaluate('model/best_13_Urban.pkl', 'database/db_esc10.sqlite', 'database/mel_esc10.hdf5')
