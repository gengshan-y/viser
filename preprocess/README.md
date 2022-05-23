## Data pre-processing

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

Please follow files in `./configs` to write the config file.

### Note on skip-frame flow
When optimizing a high framerate video, we found it useful to compute skip-frame 
optical flow to get longer-range correspondences.

After running `auto_gen.sh`, to further compute flow every 3 and 5 frames
```
bash preprocess/auto_gen_skip.sh camel
# modify $dframe variable in the script to use other skipping factors
```

Then follow `configs/camel-init.config` to set `dframe` to the number of skipped frames.
See `scripts/camel.sh` for a complete optimization pipeline.
