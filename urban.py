import numpy as np
import sys,os
import glob
import h5py
import datasetSQL
import network_arch
import librosa
import soundfile
import torch

usegpu = True
device = torch.device("cuda" if usegpu else "cpu")
torch.cuda.set_device(1)


def create_label_dataset(meta_csv, sql_file):
    ds = datasetSQL.LabelSet(sql_file)
    ds.initialize()
    print ("Creating label dataset...")

    with open(meta_csv, "r") as meta_file:
        for line in meta_file:
            audio_file,fsID,start,end,salience,fold,classID,class_name = line.split(',')
            if audio_file == "slice_file_name": #Skip the first line in CSV
                continue
            ds.__insert__({"segment_id":audio_file.strip(), "audio_file":audio_file.strip(), "fold":int(fold.strip())}, "segments")                
            ds.__insert__({"class_name":class_name.strip()}, "classes")
            #ds.__insert__({"segment_id":audio_file, "class_id":class_id, "label_type":0}, "labels")
    ds.__commit__()
    
    with open(meta_csv, "r") as meta_file:
        for line in meta_file:
            audio_file,fsID,start,end,salience,fold,classID,class_name = line.split(',')
    
            sql = """
            INSERT INTO labels (segment_id, class_id, label_type)
            SELECT '{0}', class_id, 0 FROM classes
            WHERE class_name = '{1}'""".format(audio_file, class_name.strip())
            print (sql)
            ds.cursor.execute(sql)

    ds.__commit__()
    ds.__close__()

def create_feature_dataset(audio_dir, meta_csv, h5):
    h5w = h5py.File(h5,'w')
    trg_sr = 44100
    
    with open(meta_csv, "r") as meta_file:
        for line in meta_file:
            audio_file,fsID,start,end,salience,fold,classID,class_name = line.split(',')
            if audio_file == "slice_file_name": #Skip the first line in CSV
                continue
            
            file_path = audio_dir + '/fold' + str(fold) + '/' + audio_file

            print (file_path)

            y, src_sr = soundfile.read(file_path)
                            
            if len(y.shape) > 1:
                y = y[:,0]
                
            if src_sr != trg_sr:
                y = librosa.resample(y,src_sr,trg_sr)

            mel = librosa.feature.melspectrogram(y,trg_sr,n_fft=1764,hop_length=882,n_mels=128)
            log_mel = librosa.power_to_db(mel).T
            print (log_mel.shape)
            h5w.create_dataset(audio_file, data=log_mel)

def create_embedding_dataset(feature_hdf5, embedding_hdf5, model_path):
    h5r = h5py.File(feature_hdf5,'r')
    h5w = h5py.File(embedding_hdf5,'w')
    emb_net = network_arch.EmbNet()
    emb_net.load_weight(model_path)
    emb_net.eval()
    trg_sr = 44100
    n = len([k for k in h5r.keys()])
    d = emb_net.embedding_length
    
    h5w.create_dataset('embeddings', data=np.zeros((n, d)))

    i = 0
    for k in h5r.keys():
        print (k)
        data = h5r[k][:]
        if len(data) < 201:
            data = padding(data, trg_n=201)

        torch_data = torch.from_numpy(data).float()
        with torch.no_grad():
            embedding = emb_net(torch_data.unsqueeze(0).unsqueeze(1))
            print (embedding)
        h5w.create_dataset(k, data=embedding.numpy())
        h5w['embeddings'][i] = embedding
        i += 1

def create_dist_mat(sql_file, emb_hdf5, dist_mat_hdf5):
    db = datasetSQL.LabelSet(sql_file)
    h5r = h5py.File(emb_hdf5,'r')
    h5w = h5py.File(dist_mat_hdf5,'w')
    
    d = 1024
    
    for fold_i in range(1,11):
        print (fold_i)
        sql = """
        SELECT COUNT(*) FROM segments
        WHERE fold != {0}
        """.format(fold_i)
        db.cursor.execute(sql)
        n = db.cursor.fetchone()[0]

        emb = torch.zeros((n,d), dtype=torch.float)
        sql = """
        SELECT audio_file FROM segments
        WHERE fold != {0}
        ORDER BY segment_id ASC
        """.format(fold_i)
        db.cursor.execute(sql)
        segment_list = [record[0].decode('utf-8') for record in db.cursor.fetchall()]
        #print (segment_list)
        for i,audio_file in enumerate(segment_list):
            #print (h5r[audio_file][:])
            emb[i,:] = torch.from_numpy(h5r[audio_file][:])
        print (emb.size())
        
        dist_mat = 1 - torch.mm(emb, emb.transpose(0,1))/torch.ger(torch.norm(emb, dim=1), torch.norm(emb,dim=1))
        print (dist_mat)
        h5w.create_dataset('fold{0}'.format(fold_i), data=dist_mat.cpu().numpy())

def padding(f, trg_n):
    n,d = f.shape
    r = int(trg_n/n)
    m = trg_n%n
    output = np.zeros((trg_n, d))
    for i in range(r):
        output[n*i:n*(i+1)] = f
    output[r*n:] = f[:m]
    return output


def evaluate_svm(sql_file, emb_hdf5):
    import sklearn.svm
    db = datasetSQL.LabelSet(sql_file)
    h5r = h5py.File(emb_hdf5,'r')
    d = 1024  
    
    sql = """
    DELETE FROM labels
    WHERE label_type = 3
    """
    db.cursor.execute(sql)        
    
    for fold_i in range(1,11):
        print (fold_i)
        classifier = sklearn.svm.SVC()
        """
        classifier = network_arch.ASclassifier(10)
        classifier.to(device)
        optimizer = torch.optim.Adam(classifier.parameters(), lr=5e-5)
        loss_function = torch.nn.BCELoss()
        classifier.train()
        """
        sql = """
        SELECT segments.audio_file, labels.class_id FROM segments
        INNER JOIN labels ON segments.segment_id = labels.segment_id
        WHERE segments.fold != {0} AND label_type = 0
        ORDER BY segments.segment_id ASC
        """.format(fold_i)
        db.cursor.execute(sql)
        
        example_list = [(record[0].decode('utf-8'), record[1]) for record in db.cursor.fetchall()]
        n = len(example_list)
        print (n,d)
        X = np.zeros((n,d))
        y = np.zeros(n)
        #y = np.zeros((n,10))

        for i,(k, c) in enumerate(example_list):
            X[i] = h5r[k][:]
            y[i] = c
            #y[i,c-1] = 1

        print ('training...')
        """
        torch_X = torch.from_numpy(X).float().to(device)
        torch_y = torch.from_numpy(y).float().to(device)
        
        for i_epoch in range(30000):
            output = classifier(torch_X)
            loss = loss_function(output, torch_y)
            print (loss)
            loss.backward()
            optimizer.step()
        """ 
        classifier.fit(X,y)
        
        print("Predicting...")
        #classifier.eval()
        sql = """
        SELECT segment_id, audio_file FROM segments
        WHERE fold = {0}
        ORDER BY segments.segment_id ASC
        """.format(fold_i)
        db.cursor.execute(sql)
        test_list = [(record[0].decode('utf-8'), record[1].decode('utf-8')) for record in db.cursor.fetchall()]
        n = len(test_list)
        X = np.zeros((n,d))
        for i,(segment_id, k) in enumerate(test_list):
            X[i] = h5r[k][:]
        c_list = classifier.predict(X)
        """
        with torch.no_grad():
            c_mat = classifier(torch.from_numpy(X).float().to(device))
            c_list = c_mat.argmax(dim=1) + 1
        """
        print (c_list)
        

        for i in range(n):
            sql = """
            INSERT INTO labels (segment_id, class_id, label_type)
            VALUES ('{0}', {1}, 3)
            """.format(test_list[i][0], int(c_list[i]))
            #print (sql)
            db.cursor.execute(sql)
            
        db.__commit__()
        
    sql = """
    SELECT class_id FROM labels
    WHERE label_type = 0
    ORDER BY segment_id ASC
    """
    db.cursor.execute(sql)
    ground_truth = [record[0] for record in db.cursor.fetchall()]
    
    sql = """
    SELECT class_id FROM labels
    WHERE label_type = 3
    ORDER BY segment_id ASC
    """
    db.cursor.execute(sql)
    pred = [record[0] for record in db.cursor.fetchall()]

    n = len(ground_truth)
    tp = 0
    for i in range(n):
        if pred[i] == ground_truth[i]:
            tp += 1
    print (tp*1./n)

    db.__close__()
    return
    
def evaluate_map(weight_path, db_path, feature_path):
    import embedding
    import torch
    import time
    import similarity
    import similarity_analysis
    usegpu = True
    device = torch.device("cuda" if usegpu else "cpu")
    torch.cuda.set_device(1)
    D = 1024
    
    db = datasetSQL.LabelSet(db_path)
    h5r = h5py.File(feature_path, 'r')
    
    db.cursor.execute("SELECT segment_id FROM segments WHERE audio_file NOT NULL ORDER BY segment_id ASC;")
    segment_list = [record[0].decode('utf-8') for record in db.cursor.fetchall()]
    h5w = h5py.File('/tmp/UrbanSound_emb.hdf5', 'w')
    n_segment = len(segment_list)
    h5w.create_dataset('max', (n_segment, D), dtype='float')

    emb_net = network_arch.EmbNet().to(device)
    emb_net.load_weight(weight_path)
    emb_net.eval()

    for i in range(n_segment):
        k = segment_list[i]
        print (k)
        f = h5r[k][:]

        if len(f) < 201:
            f = padding(f, trg_n=201)
        n,D = f.shape
        data = np.zeros((1,1,n,D))        
        data[0,0,:,:] = f
        torch_data = torch.from_numpy(data).float().to(device)
        t2 = time.time()
        with torch.no_grad():
            """
            pred = emb_net(torch_data)

            
            if len(pred.size()) > 2:
                embedding = torch.max(pred, dim=2)[0]
                embedding = embedding.view(embedding.size(0),-1)
                #print (embedding.shape)
            else:
                embedding = pred
            """
            embedding = emb_net(torch_data)
            print (embedding.size())
        #print (k, embedding)
        h5w['max'][i] = embedding

    h5w.close()

    h5r2 = h5py.File('/tmp/UrbanSound_emb.hdf5', 'r')
    h5w2 = h5py.File('/tmp/UrbanSound_dist.hdf5', 'w')
    similarity.Dist_gpu(h5r2, h5w2)
    h5w2.close()

    h5r3 = h5py.File('/tmp/UrbanSound_dist.hdf5', 'r')
    return (similarity_analysis.mAP2(h5r3, db))


        
def main(audio_dir, meta_csv, h5_file, sql_file):
    create_label_dataset(meta_csv, sql_file)
    #create_feature_dataset(audio_dir, meta_csv, h5_file)


                
if __name__ == '__main__':
    """
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])

    if len(sys.argv) == 4:
        h5_mel, h5_emb, model_path = sys.argv[1], sys.argv[2], sys.argv[3]
        create_embedding_dataset(h5_mel, h5_emb, model_path)

    if len(sys.argv) == 4:
        sql_file, h5_emb, h5_dist  = sys.argv[1], sys.argv[2], sys.argv[3]
        create_dist_mat(sql_file, h5_emb, h5_dist)

    if len(sys.argv) == 2:
        evaluate(sys.argv[1], 'database/db_UrbanSound.sqlite', 'database/mel_UrbanSound.hdf5')
    """
    evaluate_svm('database/db_UrbanSound8K.sqlite',  'database/emb_Urban.hdf5')
