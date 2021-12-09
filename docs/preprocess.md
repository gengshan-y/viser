## Data pre-processing (under construction, don not use)

First, download [DAVIS 2017 trainval set](https://data.vision.ee.ethz.ch/csergi/share/davis/DAVIS-2017-trainval-Full-Resolution.zip) 
and copy breakdance-flare sequence to `./database` folder.
```
cp ...davis-path/DAVIS/Annotations/Full-Resolution/breakdance-flare/ -rf database/DAVIS/Annotations/Full-Resolution/
cp ...davis-path/DAVIS-lasr/DAVIS/JPEGImages/Full-Resolution/breakdance-flare/ -rf database/DAVIS/JPEGImages/Full-Resolution/
```
Then download pre-trained VCN optical flow:
```
mkdir ./lasr_vcn
gdown https://drive.google.com/uc?id=139S6pplPvMTB-_giI6V2dxpOHGqqAdHn -O ./lasr_vcn/vcn_rob.pth
```
Run VCN-robust to predict optical flow on DAVIS breakdance-flare video:
```
bash preprocess/auto_gen.sh breakdance-flare
```
