import numpy as np
import sys,os
import glob
import h5py
import soundfile
import librosa

ESC50 = False

def create_label_dataset(meta_csv, sql_file):
    import datasetSQL
    ds = datasetSQL.LabelSet(sql_file)
    ds.initialize()
    print ("Creating label dataset...")

    with open(meta_csv, "r") as meta_file:
        for line in meta_file:
            audio_file,fold,target,category,esc10,src_file,take = line.split(',')
            if audio_file == "filename": #Skip the first line in CSV
                continue
            if ESC50 or (esc10 == "True"):
                ds.__insert__({"segment_id":audio_file, "audio_file":audio_file}, "segments")                
                ds.__insert__({"class_name":category}, "classes")
                
    with open(meta_csv, "r") as meta_file:
        for line in meta_file:
            audio_file,fold,target,category,esc10,src_file,take = line.split(',')
            if audio_file == "filename": #Skip the first line in CSV
                continue
            if ESC50 or (esc10 == "True"):
                sql = """
                INSERT INTO labels (segment_id, class_id, label_type)
                SELECT '{0}', class_id, 0 FROM classes
                WHERE class_name = '{1}'""".format(audio_file, category)
                print (sql)
                ds.cursor.execute(sql)

    ds.__commit__()
    ds.__close__()

                

def create_feature_dataset(audio_dir, meta_csv, h5):
    #Feature extraction parameters
    h5w = h5py.File(h5,'w')

    trg_sr = 44100
    #h5w.create_dataset('max', data=np.zeros((400,128)))

    with open(meta_csv, "r") as meta_file:
        for i,line in enumerate(meta_file):
            audio_file,fold,target,category,esc10,src_file,take = line.split(',')
            if audio_file == "filename": #Skip the first line in CSV
                continue
            
            if ESC50 or (esc10 == 'True'):
                print (audio_dir + '/' + audio_file)
           
                y, src_sr = soundfile.read(audio_dir + '/' + audio_file)

                if len(y.shape) > 1:
                    y = y[:,0]
                
                if src_sr != trg_sr:
                    y = librosa.resample(y,src_sr,trg_sr)

                mel = librosa.feature.melspectrogram(y,trg_sr,n_fft=1764,hop_length=882,n_mels=128)
                log_mel = librosa.power_to_db(mel).T
                print (log_mel.shape)
                #h5w['max'][i,:] = np.max(log_mel, axis=0)
                h5w.create_dataset(audio_file, data=log_mel)

                    
def main(audio_dir, meta_csv, h5_file, sql_file):
    create_feature_dataset(audio_dir, meta_csv, h5_file)
    #create_label_dataset(meta_csv, sql_file)
                
if __name__ == '__main__':
    if len(sys.argv) == 5:
        main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        print ("Argument not correct.")
