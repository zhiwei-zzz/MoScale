mkdir -p checkpoint_dir/humanml3d
cd checkpoint_dir/humanml3d

echo "Downloading evaluation models for HumanML3D dataset"
gdown --fuzzy https://drive.google.com/file/d/1sr73tfFk2O3-IL5brnZnylWi8oIVw_Hw/view?usp=drive_link
echo "Unzipping humanml3d_evaluator.zip"
unzip humanml3d_evaluator.zip
rm humanml3d_evaluator.zip

cd ../../
echo "Downloading done!"
