
import datasetSQL
import csv
import h5py
import librosa
import soundfile
import sys, os

db1_path = "database/db_train.sqlite"
feature1_path = "database/mel_train.hdf5"
db2_path = "database/db_eval.sqlite"
feature2_path = "database/mel_eval.hdf5"

file_csv = "meta/wavfiles.csv"
class_csv = "meta/class_labels_indices.csv"
wav_path = '/worktmp/zhaos/data/AudioSet'
segment1_csv =  "meta/balanced_train_segments.csv"
segment2_csv =  "meta/eval_segments.csv"


def initiate_database(db_path, segment_csv):
    db = datasetSQL.LabelSet(db_path)
    db.initialize()
    with open(segment_csv, 'r') as f:
        for item in csv.DictReader(f):
            segment = {}
            segment['segment_id'] = item['YTID']
            db.__insert__(segment, 'segments')
    db.__commit__()

def link_audio_file(db_path, filecsv):
    db = datasetSQL.LabelSet(db_path)
    with open(file_csv, 'r') as f:
        #for item in csv.DictReader(f):
        for line in f.readlines():
            item = {}
            line = line.rstrip()
            item["YTID"] = line.split(',')[-1]
            item["filename"] = ','.join(line.split(',')[:-1]) + '.wav'

            sql = """
            UPDATE segments SET audio_file = '{0}'
            WHERE segment_id = '{1}'
            """.format(item['filename'].replace("'", "''"), item['YTID'])
            db.cursor.execute(sql)
        db.__commit__()

def add_classes(db_path, class_csv):
    db = datasetSQL.LabelSet(db_path)
    with open(class_csv, 'r') as f:
        for item in csv.DictReader(f):
            class_item = {}
            class_item['ASID'] = item['mid']
            class_item['class_name'] = item['display_name'].replace("'", "''")
            print (item)
            db.__insert__(class_item, 'classes')
        db.__commit__()

def add_labels(db_path, segment_csv):
    db = datasetSQL.LabelSet(db_path)
    with open(segment_csv, 'r') as f:
        for line in f.readlines():
            segment_id = line.split(',')[0]
            if segment_id == "YTID": #title line
                continue
            labels = ','.join(line.split(',')[3:]).replace('"', '').strip()
            for label in labels.split(','):
                sql = """
                INSERT INTO labels (segment_id, class_id, label_type)                
                SELECT segments.segment_id, classes.class_id, 0 FROM segments CROSS JOIN classes WHERE segments.segment_id = '{0}' AND classes.ASID = '{1}'
                """.format(segment_id, label)
                print (sql)
                db.cursor.execute(sql)
    db.__commit__()

            
            #item['positive_labels'] = ','.join(line.split(',')[3:])
            
            #print (item["positive_labels"])
            
def compute_features(db_path, feature_path, wav_root):
    h5w = h5py.File(feature_path, 'w')
    db = datasetSQL.LabelSet(db_path)
    trg_sr = 32000
    
    sql = """
    SELECT segment_id, audio_file FROM segments
    WHERE audio_file NOT NULL
    """
    segment_list = db.cursor.execute(sql)
    for segment_tuple in segment_list:
        segment_id, audio_file = segment_tuple[0].decode('utf-8'), segment_tuple[1].decode('utf-8')
        print (segment_id, audio_file)
        y, src_sr = soundfile.read(os.path.join(wav_root, audio_file))
        if len(y.shape) > 1:
            y = y[:,0]
        y = librosa.core.resample(y, src_sr, trg_sr)
        mel = librosa.feature.melspectrogram(y,trg_sr,n_fft=1024,hop_length=512,n_mels=128)
        log_mel = librosa.power_to_db(mel).T
        print (log_mel.shape)
        h5w.create_dataset(segment_id, data=log_mel)
    return


if __name__ == '__main__':
    initiate_database(db2_path, segment2_csv)
    link_audio_file(db2_path, file_csv)
    add_classes(db2_path, class_csv)
    add_labels(db2_path, segment2_csv)
    compute_features(db2_path, feature2_path, wav_path)
