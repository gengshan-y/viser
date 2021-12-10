dev=0
ngpu=1
batch_size=4

#dev=0,1,2,3
#ngpu=4
#batch_size=1

seed=1003
address=1111
logname=breakdance-flare-$seed
checkpoint_dir=log

# optimize viser on a subset of video frames
# for breakdance-flare we use start: 22, end: 42
dataname=breakdance-flare-init
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-0 --checkpoint_dir $checkpoint_dir --n_bones 21 \
    --num_epochs 20 --dataname $dataname --ngpu $ngpu --batch_size $batch_size --seed $seed
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-1 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs 10 --dataname $dataname  --ngpu $ngpu --batch_size $batch_size \
    --model_path $checkpoint_dir/$logname-0/pred_net_latest.pth --finetune --n_faces 1601
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-2 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs 10 --dataname $dataname  --ngpu $ngpu --batch_size $batch_size \
    --model_path $checkpoint_dir/$logname-1/pred_net_latest.pth  --finetune --n_faces 1602

# start-idx and end-idx are determined by the initization subset of frames
# delta-max-cap is computed as max( number-of-frames - end-idx, start-idx - 0)
dataname=breakdance-flare
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-ft1 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs 60 --dataname $dataname --ngpu $ngpu --batch_size $batch_size \
    --model_path $checkpoint_dir/$logname-2/pred_net_latest.pth --finetune --n_faces 1601 \
    --start_idx 22 --end_idx 42 --use_inc --delta_max_cap 30 
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-ft2 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs 20 --dataname $dataname  --ngpu $ngpu --batch_size $batch_size \
    --model_path $checkpoint_dir/$logname-ft1/pred_net_latest.pth --finetune --n_faces 8000
