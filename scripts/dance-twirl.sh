dev=0
ngpu=1
batch_size=4

#dev=0,1,2,3
#ngpu=4
#batch_size=1

seed=1003
address=1112
logname=dance-twirl-$seed
checkpoint_dir=log

dataname=dance-twirl-init
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-0 --checkpoint_dir $checkpoint_dir --n_bones 21 \
    --num_epochs 20 --dataname $dataname --ngpu $ngpu --batch_size $batch_size --seed $seed \
    --flow_wt 1
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-1 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs 10 --dataname $dataname  --ngpu $ngpu --batch_size $batch_size \
    --model_path $checkpoint_dir/$logname-0/pred_net_latest.pth --finetune --n_faces 1601 \
    --flow_wt 1
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-2 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs 10 --dataname $dataname  --ngpu $ngpu --batch_size $batch_size \
    --model_path $checkpoint_dir/$logname-1/pred_net_latest.pth  --finetune --n_faces 1602 \
    --flow_wt 1

dataname=dance-twirl
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-ft1 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs 60 --dataname $dataname --ngpu $ngpu --batch_size $batch_size \
    --model_path $checkpoint_dir/$logname-2/pred_net_latest.pth --finetune --n_faces 1601 \
    --start_idx 24 --end_idx 40 --use_inc --delta_max_cap 50 \
    --flow_wt 1
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-ft2 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs 20 --dataname $dataname  --ngpu $ngpu --batch_size $batch_size \
    --model_path $checkpoint_dir/$logname-ft1/pred_net_latest.pth --finetune --n_faces 8000
