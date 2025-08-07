import numpy as np
from scipy.io import wavfile
from scipy.signal import resample
from mel_cepstral_distance import compare_audio_files

def resample_audio(audio, orig_sr, target_sr):
    if orig_sr == target_sr:
        return audio
    duration = len(audio) / orig_sr
    target_length = int(duration * target_sr)
    return resample(audio, target_length)

# Assume files are in current directory
ground_truth_path = 'ground_truth.wav'  # Replace with actual filename
own_model_path = 'sample_gen.wav'      # Replace with actual filename
pretrained_model_path = 'pretrained_model.wav'  # Replace with actual filename
target_sr = 22050

# Load audios
sr_gt, gt_audio = wavfile.read(ground_truth_path)
sr_own, own_audio = wavfile.read(own_model_path)
sr_pre, pre_audio = wavfile.read(pretrained_model_path)

# Handle stereo audio by selecting one channel
if gt_audio.ndim > 1:
    gt_audio = gt_audio[:, 0]
if own_audio.ndim > 1:
    own_audio = own_audio[:, 0]
if pre_audio.ndim > 1:
    pre_audio = pre_audio[:, 0]

# Resample to target sample rate
gt_audio = resample_audio(gt_audio, sr_gt, target_sr)
own_audio = resample_audio(own_audio, sr_own, target_sr)
pre_audio = resample_audio(pre_audio, sr_pre, target_sr)

# Normalize to float32 [-1, 1]
gt_audio = gt_audio.astype(np.float32) / np.max(np.abs(gt_audio))
own_audio = own_audio.astype(np.float32) / np.max(np.abs(own_audio))
pre_audio = pre_audio.astype(np.float32) / np.max(np.abs(pre_audio))

# Save temporary files for mel-cepstral-distance
np.save('temp_gt.npy', gt_audio)
np.save('temp_own.npy', own_audio)
np.save('temp_pre.npy', pre_audio)
wavfile.write('temp_gt.wav', target_sr, gt_audio)
wavfile.write('temp_own.wav', target_sr, own_audio)
wavfile.write('temp_pre.wav', target_sr, pre_audio)

# Compute MCD using mel-cepstral-distance
mcd_own, penalty_own = compare_audio_files('temp_gt.wav', 'temp_own.wav')
mcd_pre, penalty_pre = compare_audio_files('temp_gt.wav', 'temp_pre.wav')

# Clean up temporary files
import os
os.remove('temp_gt.wav')
os.remove('temp_own.wav')
os.remove('temp_pre.wav')
os.remove('temp_gt.npy')
os.remove('temp_own.npy')
os.remove('temp_pre.npy')

print(f"MCD for own model: {mcd_own:.2f} (Penalty: {penalty_own:.4f})")
print(f"MCD for pretrained model: {mcd_pre:.2f} (Penalty: {penalty_pre:.4f})")