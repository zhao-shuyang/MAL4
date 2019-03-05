import sys
import csv
import subprocess

audio_folder = "/proj/asignal/Audioset/wav"
train_csv = 'meta/balanced_train_segments.csv'
test_csv = 'meta/eval_segments.csv'

def extract(meta_file):
    with open(meta_file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for item in reader:
            print(item)
            command = """
            youtube-dl -f bestaudio https://www.youtube.com/watch?v=%s --exec "ffmpeg -y -ss %s -i {} -acodec %s -ar %d -ac 1 -t 10 %s/{}.wav && rm {} && echo {},%s >> meta/wavfiles2.csv;";
            """%(item['YTID'], item['start_seconds'],"pcm_s16le", 44100, audio_folder, item['YTID'])
            print (command)
            subprocess.call(command, shell=True)


if __name__ == '__main__':
    #extract(train_csv)
    extract(test_csv)
