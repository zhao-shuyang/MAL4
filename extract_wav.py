import sys
import csv
import subprocess

audio_folder = "/home/zhaos/Audioset_wav"

with open('meta/balanced_train_segments.csv', 'r') as csvfile:
    reader = csv.DictReader(csvfile)
    for item in reader:
        print(item)
        command = """
        youtube-dl -f bestaudio https://www.youtube.com/watch?v=%s --exec "ffmpeg -y -ss %s -i {} -acodec %s -ar %d -ac 1 -t 10 %s/{}.wav && rm {} && echo {},%s >> meta/wavfiles.csv;";
        """%(item['YTID'], item['start_seconds'],"pcm_s16le", 44100, audio_folder, item['YTID'])
        subprocess.call(command, shell=True)
