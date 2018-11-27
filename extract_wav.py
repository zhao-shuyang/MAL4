import sys
import csv
import subprocess

audio_folder = "wav"

with open('meta/balanced_train_segments.csv', 'rb') as csvfile:
    reader = csv.DictReader(csvfile)
    for item in reader:
        print(item)
        command = """
        youtube-dl https://www.youtube.com/watch?v=%s --exec "ffmpeg -y -ss %s -i {} -acodec %s -ar %d -ac 1 -t 10 %s/{}.wav; rm {};"
        """%(item['YTID'], item['start_seconds'],"pcm_s16le", 44100, audio_folder)
        subprocess.call(command, shell=True)
