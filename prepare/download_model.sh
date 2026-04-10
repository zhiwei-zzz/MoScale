mkdir -p checkpoint_dir/humanml3d
cd checkpoint_dir/humanml3d

echo "Downloading checkpoints"
gdown --fuzzy https://drive.google.com/file/d/1Cb9bSWBsLbjHVYO0UnN6MBAe-WChSUpz/view?usp=drive_link
echo "Unzipping MoScale_checkpoint"
unzip MoScale_checkpoint
rm MoScale_checkpoint.zip

cd ../../
echo "Downloading done!"
