# ViSER

## Installation with conda
```
conda env create -f viser.yml
conda activate viser-release
# install softras
cd third_party/softras; python setup.py install; cd -;
# install manifold remeshing
git clone --recursive git://github.com/hjwdzh/Manifold; cd Manifold; mkdir build; cd build; cmake .. -DCMAKE_BUILD_TYPE=Release;make -j8; cd ../../
```

## Data preparation
Create folders to store intermediate data and training logs
```
mkdir log; mkdir tmp; 
```
Download pre-processed data (rgb, mask, flow) following the link 
[here](https://www.dropbox.com/sh/3l02a34w0wk34gs/AAD0WUJIayFUYNVg8LZOP6gMa?dl=0) 
and unzip under `./database/DAVIS/`. The dataset is organized as:
```
DAVIS/
    Annotations/
        Full-Resolution/
            sequence-name/
                {%05d}.png
    JPEGImages/
        Full-Resolution/
            sequence-name/
                {%05d}.jpg
    FlowBW/ and FlowFw/
        Full-Resolution/
            sequence-name/ and optionally seqname-name_{%02d}/ (frame interval)
                flo-{%05d}.pfm
                occ-{%05d}.pfm
                visflo-{%05d}.jpg
                warp-{%05d}.jpg
```

To run preprocessing scripts on other videos, see [install.md](./docs/preprocess.md). 

## Example: breakdance-flare
Run 
```
bash scripts/template.sh breakdance-flare
```

To monitor optimization, run
```
tensorboard --logdir log/
```

To render optimized breakdance-flare
```
bash scripts/render_result.sh breakdance-flare log/breakdance-flare-1003-ft2/pred_net_20.pth 36
```


Example outputs:

<p align="center"> 
<img src="figs/rbreakdance-flare-all.gif" alt="" width="66.7%" />
</p>

## Example: elephants (under construction)
Run
```
bash scripts/relephant-walk.sh
```

To monitor optimization, run
```
tensorboard --logdir log/
```

To render optimized breakdance-flare
```
bash scripts/render_elephants.sh log/elephant-walk-1003-6/pred_net_10.pth
```

## Additional Notes

<details><summary>Distributed training</summary>

The current codebase supports single-node multi-gpu training with pytorch distributed data-parallel.
Please modify `dev` and `ngpu` in `scripts/template.sh` to select devices.

</details>

<details><summary>Potential bugs</summary>

- When setting batch_size to 3, rendered flow may become constant values.
</details>

## Acknowledgement

The code borrows the skeleton of [CMR](https://github.com/akanazawa/cmr)

External repos:
- [SoftRas](https://github.com/ShichenLiu/SoftRas)
- [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity)
- [Manifold](https://github.com/hjwdzh/Manifold)
- [VCN-robust](https://github.com/gengshan-y/rigidmask)
- [Nerf-pytorch](https://github.com/krrish94/nerf-pytorch)

## Citation

<details><summary>To cite our paper</summary>

```
@inproceedings{yang2021viser,
  title={ViSER: Video-Specific Surface Embeddings for Articulated 3D Shape Reconstruction},
  author={Yang, Gengshan 
      and Sun, Deqing
      and Jampani, Varun
      and Vlasic, Daniel
      and Cole, Forrester
      and Liu, Ce
      and Ramanan, Deva},
  booktitle = {NeurIPS},
  year={2021}
}  
```
```
@inproceedings{yang2021lasr,
  title={LASR: Learning Articulated Shape Reconstruction from a Monocular Video},
  author={Yang, Gengshan 
      and Sun, Deqing
      and Jampani, Varun
      and Vlasic, Daniel
      and Cole, Forrester
      and Chang, Huiwen
      and Ramanan, Deva
      and Freeman, William T
      and Liu, Ce},
  booktitle={CVPR},
  year={2021}
}  
```
</details>

## TODO
- data pre-processing scripts
- evaluation data and scripts
- code clean up
