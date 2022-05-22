dev=0
ngpu=1
bs=4

#dev=0,1,2,3
#ngpu=4
#bs=1

seed=1003
address=1259
logname=camel-$seed
checkpoint_dir=log
nepoch=10

dataname=camel-init
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-0 --checkpoint_dir $checkpoint_dir --n_bones 21 \
    --num_epochs 20 --dataname $dataname --ngpu $ngpu --batch_size $bs --cnnpp --seed $seed
dataname=camel
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-1 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs $nepoch --dataname $dataname  --ngpu $ngpu --batch_size $bs -cnnpp \
    --model_path $checkpoint_dir/$logname-0/pred_net_latest.pth   --finetune --n_faces 1601 
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-2 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs $nepoch --dataname $dataname  --ngpu $ngpu --batch_size $bs -cnnpp \
    --model_path $checkpoint_dir/$logname-1/pred_net_latest.pth   --finetune --n_faces 1602 
CUDA_VISIBLE_DEVICES=$dev python -m torch.distributed.launch \
    --master_port $address --nproc_per_node=$ngpu optimize.py \
    --name=$logname-3 --checkpoint_dir $checkpoint_dir --n_bones 36 \
    --num_epochs $nepoch --dataname $dataname  --ngpu $ngpu --batch_size $bs -cnnpp \
    --model_path $checkpoint_dir/$logname-2/pred_net_latest.pth   --finetune --n_faces 1603
