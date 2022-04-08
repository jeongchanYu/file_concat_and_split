import os
import librosa
import csv
import soundfile as sf


concatenated_time_data_csv_path = './output/concat/man/man_concatenated_time_data.csv'
concatenated_wav_path = './output/concat/man/man_concatenated.wav'
split_wav_path = './output/split/man'

sampling_rate = 16000

x, sr = librosa.load(concatenated_wav_path, sr=None)
if sr != sampling_rate:
    raise Exception(f"Different sampling rate detected ! -> {concatenated_wav_path} ({sr})")

with open(concatenated_time_data_csv_path, 'r') as f:
    csv_list = list(csv.reader(f))[1:]
    for file_name, start, end in csv_list:
        save_path = os.path.join(split_wav_path, file_name)
        split_x = x[int(start):int(end)]

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        sf.write(save_path, split_x, sampling_rate)