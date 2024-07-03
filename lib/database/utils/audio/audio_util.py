import numpy as np
from .vggish_src import mel_features
from .vggish_src import vggish_params
import torch, resampy, librosa


def audio_to_mel(audio, sr, n_mels=64, n_fft=2048, hop_length=256):
    target_sr = 16000
    if sr != target_sr:
        audio = resampy.resample(audio, sr, target_sr)
            
    # Compute the mel spectrogram
    mel_spec = librosa.feature.melspectrogram(y=audio, sr=target_sr, 
                                              n_fft=n_fft, 
                                              hop_length=hop_length, 
                                              n_mels=n_mels)
    
    # Convert to decibels (log scale)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    
    return mel_spec_db
  
  
def waveform_to_examples(data, sample_rate, return_tensor=True):
    """Converts audio waveform into an array of examples for VGGish.

  Args:
    data: np.array of either one dimension (mono) or two dimensions
      (multi-channel, with the outer dimension representing channels).
      Each sample is generally expected to lie in the range [-1.0, +1.0],
      although this is not required.
    sample_rate: Sample rate of data.
    return_tensor: Return data as a Pytorch tensor ready for VGGish

  Returns:
    3-D np.array of shape [num_examples, num_frames, num_bands] which represents
    a sequence of examples, each of which contains a patch of log mel
    spectrogram, covering num_frames frames of audio and num_bands mel frequency
    bands, where the frame length is vggish_params.STFT_HOP_LENGTH_SECONDS.

  """
    # Convert to mono.
        
    # Resample to the rate assumed by VGGish.
    if sample_rate != vggish_params.SAMPLE_RATE:
        data = resampy.resample(data, sample_rate, vggish_params.SAMPLE_RATE)

    # Compute log mel spectrogram features.
    log_mel = mel_features.log_mel_spectrogram(
        data,
        audio_sample_rate=vggish_params.SAMPLE_RATE,
        log_offset=vggish_params.LOG_OFFSET,
        window_length_secs=vggish_params.STFT_WINDOW_LENGTH_SECONDS,
        hop_length_secs=vggish_params.STFT_HOP_LENGTH_SECONDS,
        num_mel_bins=vggish_params.NUM_MEL_BINS,
        lower_edge_hertz=vggish_params.MEL_MIN_HZ,
        upper_edge_hertz=vggish_params.MEL_MAX_HZ)

    # Frame features into examples.
    features_sample_rate = 1.0 / vggish_params.STFT_HOP_LENGTH_SECONDS
    example_window_length = int(round(
        vggish_params.EXAMPLE_WINDOW_SECONDS * features_sample_rate))
    example_hop_length = int(round(
        vggish_params.EXAMPLE_HOP_SECONDS * features_sample_rate))
    
    log_mel_examples = mel_features.frame(
        log_mel,
        window_length=example_window_length,
        hop_length=example_hop_length)

    if return_tensor:
        log_mel_examples = torch.tensor(
            log_mel_examples, requires_grad=True)[:, None, :, :].float()

    return log_mel_examples
