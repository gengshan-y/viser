davisdir=./database/DAVIS/
res=Full-Resolution
seqname=$1
newname=r${seqname}
dframe=(3 5) # e.g., pass (2 4 8) to compute flow between every 2 4 8 frame
testres=1

# run skip frame flow estimation (optional, useful for high frame frame videos such as elephant-walk and camel)
for i in "${dframe[@]}"
do
CUDA_VISIBLE_DEVICES=1 python preprocess/skip_gen.py --datapath $davisdir/JPEGImages/$res/$newname/ \
    --loadmodel ./lasr_vcn/vcn_rob.pth  --testres $testres --dframe $i
done
