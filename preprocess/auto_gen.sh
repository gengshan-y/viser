davisdir=./database/DAVIS/
res=Full-Resolution
seqname=$1
newname=r${seqname}
array=(2 4 8)
testres=1

# run flow estimation
CUDA_VISIBLE_DEVICES=0 python preprocess/auto_gen.py --datapath $davisdir/JPEGImages/$res/$seqname/ \
    --loadmodel ./lasr_vcn/vcn_rob.pth  \
    --testres $testres --flow_threshold 0

mkdir -p $davisdir/JPEGImages/$res/$newname
mkdir -p $davisdir/Annotations/$res/$newname
mkdir -p $davisdir/FlowFW/$res/$newname
mkdir -p $davisdir/FlowBW/$res/$newname
cp $seqname/JPEGImages/*   -rf $davisdir/JPEGImages/$res/$newname
cp $seqname/Annotations/* -rf $davisdir/Annotations/$res/$newname
cp $seqname/FlowFW/*           -rf $davisdir/FlowFW/$res/$newname
cp $seqname/FlowBW/*           -rf $davisdir/FlowBW/$res/$newname

## run skip frame flow estimation (optional, useful for high frame frame videos such as elephant-walk)
#for i in "${array[@]}"
#do
#CUDA_VISIBLE_DEVICES=1 python preprocess/skip_gen.py --datapath $davisdir/JPEGImages/$res/$newname/ \
#    --loadmodel ./lasr_vcn/vcn_rob.pth  --testres $testres --dframe $i
#done
