def fix_len_compatibility(length, num_downsamplings_in_unet=4):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1
# data parameters

valid_filelist_path = None
test_filelist_path = None
n_feats = 80
n_spks = 247  # 247 for Libri-TTS filelist and 1 for LJSpeech
spk_emb_dim = 192
decoder_spk_emb_dim = 64
n_fft = 1024
sample_rate = 22050

hop_length = 256
win_length = 1024
f_min = 0
f_max = 8000

# encoder parameters
n_enc_channels = 192
filter_channels = 768
filter_channels_dp = 256
n_enc_layers = 6
enc_kernel = 3
enc_dropout = 0.1
n_heads = 2
window_size = 4
dec_dim = 384
# decoder parameters

beta_min = 1e-4
beta_max = 0.05
pe_scale = 1000  # 1 for `grad-tts-old.pt` checkpoint
d_phoneme=768
d_speaker=192
# training parameters
log_dir = 'logs/new_exp'
test_size = 4
n_epochs = 200
batch_size = 32
learning_rate = 1e-4
seed = 37
save_every = 8
out_size = fix_len_compatibility(2*22050//256)

n_timesteps=200