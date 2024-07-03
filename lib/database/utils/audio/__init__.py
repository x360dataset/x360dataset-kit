from .audio_util import waveform_to_examples, audio_to_mel
import librosa
import numpy as np


def read_at(self, mp4file, max_len=250):
    name = mp4file.split("/")[-1]
    at = np.load(mp4file.replace(name, "at.npy"))[:max_len]
    if len(at) < 255:
        at = np.pad(at, (0, 255 - len(at)), 'constant', constant_values=(0, 0))
    return at
    
# .mp4 -> .wav -> mel
def read_audio(self, path, sr=16000, speed=1, offset=0, fps=25, 
               frame_num=25 * 10, max_len = (129, 80), mode="train",
               au_transform=None, spec_transform=None, use_fourier=False):

    sr = sr // speed  # speed up the audio by "speed rate"
    if ".mp4" in path:
        path = path.replace(".mp4", ".wav")

    offset = 0 if mode != "train" else offset   # follow the offset from video
    
    sample, _ = librosa.load(path, offset=0.0, mono=False, sr=sr)  # sr=16000 - > 220618
    left   = sample[0]
    right  = sample[1]
    length = len(left)
    
    # Audio sample
    start = np.minimum(int(sr * offset/fps), length - sr)    
    end   = np.minimum(int(sr * (offset+frame_num)/fps), len(left))


    left  = left[start:end]
    right = right[start:end] 
        

    
    if use_fourier:
        # Fourier Transform
        left  = process_audio_sample(left, offset/frame_num, sr, au_transform, spec_transform) 
        right = process_audio_sample(right, offset / frame_num, sr, au_transform, spec_transform)


        left = left[:max_len[0], :max_len[1]]
        right = right[:max_len[0], :max_len[1]]


        if left.shape[1] < max_len[1] or left.shape[0] < max_len[0]:
            offset0 = np.maximum(max_len[0] - left.shape[0], 0)
            offset1 = np.maximum(max_len[1] - left.shape[1], 0)

            left = np.pad(left, ((0, offset0), (0, offset1)),
                            'constant', constant_values=(0, 0))
            right = np.pad(right, ((0, offset0), (0, offset1)),
                            'constant', constant_values=(0, 0))

    else:
        # Mel Spectrogram
        n_mels = 50
        n_fft = 512 # 2048
        hop_length = 64
        right = audio_to_mel(right, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)
        left  = audio_to_mel(left, sr, n_mels=n_mels, n_fft=n_fft, hop_length=hop_length)

        target_mel = 5
        target_l2 = 64
        l1 = 50
        l2 = 160

        right = right[:l1, :l2]
        left = left[:l1, :l2]
        
        offset0 = np.maximum(l1 - left.shape[0], 0)
        offset1 = np.maximum(l2 - left.shape[1], 0)

        left = np.pad(left, ((0, offset0), (0, offset1)), 'constant', constant_values=(0, 0))
        right = np.pad(right, ((0, offset0), (0, offset1)), 'constant', constant_values=(0, 0)) 
        
        left = left.reshape((target_mel, -1, target_l2))
        right = right.reshape((target_mel, -1, target_l2))

    return [left, right]




def process_audio_sample(self, sample, frame_offset_ratio, rate=16000,
                            au_transform=None, spec_transform=None, fps=25):

        if au_transform:
            sample = au_transform(samples=sample, sample_rate=rate)
        # Norm the sample
        sample = np.clip(sample, -1, 1)

        spectrogram = librosa.stft(sample, n_fft=256, hop_length=128)
        spectrogram = np.log(np.abs(spectrogram) + 1e-3)

        if spec_transform:
            spectrogram = spec_transform(spectrogram)

        return spectrogram