import os
import glob
import librosa
import numpy as np
import soundfile as sf


wav_file_for_concat_path = './dataset/man'
concatenated_time_data_csv_path = './output/concat/man/man_concatenated_time_data.csv'
concatenated_wav_path = './output/concat/man/man_concatenated.wav'

sampling_rate = 16000
zero_padding_time = 0.1


wav_file_list = sorted(list(map(lambda path: os.path.normpath(os.path.abspath(path)) , glob.glob(wav_file_for_concat_path + '/**/*.wav', recursive=True))))


zero_padding_sample = np.zeros(int(zero_padding_time*sampling_rate))


concat_list = [zero_padding_sample]
start_sample_index = len(zero_padding_sample)

os.makedirs(os.path.dirname(concatenated_time_data_csv_path), exist_ok=True)
with open(concatenated_time_data_csv_path, 'w') as f:
    f.write("file_name,start,end\n")
    for wav_path in wav_file_list:
        x, sr = librosa.load(wav_path, sr=None)
        if sr != sampling_rate:
            raise Exception(f"Different sampling rate detected ! -> {wav_path} ({sr})")

        concat_list.append(x)
        concat_list.append(zero_padding_sample)

        end_sample_index = start_sample_index + len(x)

        # save csv
        wav_path_for_csv = wav_path.replace(os.path.normpath(os.path.abspath(wav_file_for_concat_path)), "").lstrip('\/.')
        f.write(f'{wav_path_for_csv},{start_sample_index},{end_sample_index}\n')

        start_sample_index = end_sample_index + len(zero_padding_sample)

concat_list.append(zero_padding_sample)
concatenated_wav_data = np.concatenate(concat_list)

os.makedirs(os.path.dirname(concatenated_wav_path), exist_ok=True)
sf.write(concatenated_wav_path, concatenated_wav_data, sampling_rate)