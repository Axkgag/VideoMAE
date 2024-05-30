# Set the path to save video
OUTPUT_DIR='./TODO/results/kinetics'
# path to video for visualization
VIDEO_PATH='./TODO/videos/k4_test.mp4'
# path to pretrain model
MODEL_PATH='./TODO/models/videomae_s_k4.pth'

python run_videomae_vis.py --mask_ratio 0.9 --mask_type tube --decoder_depth 4 --model pretrain_videomae_small_patch16_224 ${VIDEO_PATH} ${OUTPUT_DIR} ${MODEL_PATH}

# C: > ProgramData > Anaconda3 > envs > pytorch > Lib > site-packages > timm > models >  factory.py
# def create_model
# line71 -> line72