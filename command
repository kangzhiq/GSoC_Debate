srun -p gpu -C gpup100 --gres=gpu:1 --mem=8gb --pty bash

singularity shell --bind /mnt --nv openpose-zhiqi-cuda

./build/examples/openpose/openpose.bin -image_dir /mnt/rds/redhen/gallina/home/zxk93/debate/Frames_fps15/ -render_pose=0 -display=0 -write_json /mnt/rds/redhen/gallina/home/zxk93/debate/json_fps15/
