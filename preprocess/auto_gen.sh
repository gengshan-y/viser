# Copyright 2021 Google LLC
# 
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
#     https://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


davisdir=./database/DAVIS/
res=Full-Resolution
seqname=$1
newname=r${seqname}
array=(2 4 8)
testres=1

rm ./$seqname -rf
# run flow estimation
CUDA_VISIBLE_DEVICES=1 python auto_gen.py --datapath $indavisdir/JPEGImages/$res/$seqname/ --loadmodel ../tmp/vcn_rob.pth  --testres $testres --medflow 0

mkdir $davisdir/JPEGImages/$res/$newname
mkdir $davisdir/Annotations/$res/$newname
mkdir $davisdir/FlowFW/$res/$newname
mkdir $davisdir/FlowBW/$res/$newname
cp $seqname/JPEGImages/*   -rf $davisdir/JPEGImages/$res/$newname
cp $seqname/Annotations/* -rf $davisdir/Annotations/$res/$newname
cp $seqname/FlowFW/*           -rf $davisdir/FlowFW/$res/$newname
cp $seqname/FlowBW/*           -rf $davisdir/FlowBW/$res/$newname
rm ./$seqname -rf


for i in "${array[@]}"
do
CUDA_VISIBLE_DEVICES=1 python skip_gen.py --datapath $outdavisdir/JPEGImages/$res/$newname/ --loadmodel ../tmp/vcn_rob.pth  --testres $testres --dframe $i
done

