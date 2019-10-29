import librosa
import pdb
import librosa.display
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import kaldi


wavpath="/lan/ibdata/SPEECH_DATABASE/voxceleb1/train/wav/id10001/1zcIwhmdeo4/00001.wav"
y, sr = librosa.load(wavpath, sr=None)
hop_dur = 0.01
frame_dur = 0.025
S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, hop_length=int(hop_dur*sr), n_fft=int(frame_dur*sr), fmax=8000)
fig = plt.figure(figsize=(10, 4))
librosa.power_to_db(S, ref=np.max)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), y_axis='mel', fmax=8000, x_axis='time')
plt.colorbar(format='%+2.0f dB')
plt.title('Mel spectrogram')
plt.tight_layout()
fig.savefig('test.png')
