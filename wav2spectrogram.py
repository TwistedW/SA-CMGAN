import librosa
import librosa.display
import matplotlib.pyplot as plt
import os, glob
from tqdm import tqdm
import numpy as np

wavpath = "./dataset/audio"
imgpath = "./dataset/spectrogram"

audio_classs = ['fou', 'hou', 'qian', 'shi', 'you', 'zuo']
for i in range(len(audio_classs)):
    audios = os.listdir(wavpath + '/' + audio_classs[i])
    new_audios = []
    for k in range(len(audios)):
        if '._new' in audios[k]:
            pass
        else:
            new_audios.append(audios[k])
    audios = new_audios
    print(audios)
    for j in tqdm(range(len(audios))):
        audio_path = wavpath+'/'+audio_classs[i]+'/'+audios[j]
        # Load a wav file
        name, ext = os.path.splitext(audios[j])
        y, sr = librosa.load(audio_path, sr=None)
        # extract mel spectrogram feature
        melspec = librosa.feature.melspectrogram(y, sr, n_fft=2048, hop_length=512, n_mels=128)
        # convert to log scale
        logmelspec = librosa.amplitude_to_db(melspec, ref=np.max)
        #logmelspec = librosa.power_to_db(melspec, ref=np.max)
        plt.figure()
        librosa.display.specshow(logmelspec)
        save_path = imgpath+'/'+audio_classs[i]
        if not os.path.exists(save_path):
            os.mkdir(save_path)
        plt.savefig(save_path+'/' + name + ".png")
        plt.close()

