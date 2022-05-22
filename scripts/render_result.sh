seqname=$1
model_path=$2
n_bones=$3

testdir=${model_path%/*} # %: from end
add_args=${*: 3:$#-1}
echo $add_args
suffix=.mp4
prefix=$testdir/$seqname

# predict articulated meshes
python extract.py --model_path $model_path  --checkpoint_dir $testdir\
            --dataname $seqname --n_bones $n_bones \
            $add_args

# convert to videos
ffmpeg -r 7 -i $testdir/render-%*.png -c:v libx264 -vf fps=$fr -pix_fmt  \
        yuv420p $testdir/$seqname-$epoch.mp4
ffmpeg \
  -i $testdir/$seqname-$epoch.mp4 \
     $testdir/$seqname-$epoch.gif
sleep 1

# image
python render_vis.py --testdir $testdir --outpath $prefix-vid$suffix --seqname $seqname --freeze no --overlay no --fixlight no --vis_bone yes --append_img yes --append_render no
sleep 1
# reconstruction vp1
python render_vis.py --testdir $testdir --outpath $prefix-vp1$suffix --seqname $seqname --freeze no --overlay no --watertight yes --fixlight no --vis_bone no --append_img no --append_render yes
sleep 1
# texture
python render_vis.py --testdir $testdir --outpath $prefix-tex$suffix --seqname $seqname --freeze no --overlay no --fixlight no --vis_bone no --append_img no --append_render yes
sleep 1
# reconstruction vp2
python render_vis.py --testdir $testdir --outpath $prefix-vp2$suffix --seqname $seqname --freeze no --overlay no --watertight yes --fixlight no --vis_bone no --append_img no --append_render yes --vp2
sleep 1
# reconstruction vp3
python render_vis.py --testdir $testdir --outpath $prefix-vp3$suffix --seqname $seqname --freeze no --overlay no --watertight yes --fixlight no --vis_bone no --append_img no --append_render yes --vp3
sleep 1
# reconstruction turntable
python render_vis.py --testdir $testdir --outpath $prefix-sta$suffix --seqname $seqname --freeze yes --overlay no --fixlight no --vis_bone no --append_img no --append_render yes
sleep 1
# bones
python render_vis.py --testdir $testdir --outpath $prefix-bne$suffix --seqname $seqname --freeze no --overlay no --fixlight no --vis_bone yes --append_img no --append_render yes
sleep 1
# parts
python render_vis.py --testdir $testdir --outpath $prefix-cor$suffix --seqname $seqname --freeze no --overlay no --fixlight no --vis_bone no --append_img no --append_render yes --corresp
sleep 1
# trajectory
python render_vis.py --testdir $testdir --outpath $prefix-tra$suffix --seqname $seqname --freeze no --overlay no --fixlight no --vis_bone no --append_img no --append_render yes --vis_traj yes
sleep 1

ffmpeg -y  -i $prefix-vid$suffix \
           -i $prefix-tex$suffix \
           -i $prefix-cor$suffix \
           -i $prefix-bne$suffix \
           -i $prefix-vp1$suffix \
           -i $prefix-vp2$suffix \
           -i $prefix-vp3$suffix \
           -i $prefix-tra$suffix \
-filter_complex "[0:v][1:v][2:v][3:v]hstack=inputs=4[top];\
[4:v][5:v][6:v][7:v]hstack=inputs=4[bottom];\
[top][bottom]vstack=inputs=2[v]" \
-map "[v]" \
$prefix-all$suffix

ffmpeg -y -i $prefix-all.mp4 -vf "scale=iw/2:ih/2" $prefix-all.gif
