this is for course project submission, model download link:https://drive.google.com/file/d/1gUqYsrUeWjk0uS7LIGd_6JhSwW5LV1Yu/view?usp=sharing

f_data_stage2.py is the dataset class for loading data during training
f_params2.py is training and model parameter
f_stage2SpeakerEmbeddinggenerator.py for collecting speaker embedding and store it in data/LibriTTS-R/speaker_embeddings_192_r_aligned.pt
f_tiny_diffusion.py is the model structure
goodsample folder has some examples of model generation
checkmel.py is used to check mel during training to generate output_comparison.png
check.py is used for evaluation-MCD score, compare pretrained_model.wav,own_model.wav,ground_truth.wav
verify.py is for test
f_stage2filelistgenerator.py is used to merge speaker embedding, duration information, and phoneme information to generate a dataset and make phoneme dictionary and filelist in data/LibriTTS-R/full_libritts_aligned_filelist.txt         data/LibriTTS-R/phone2idx.json

