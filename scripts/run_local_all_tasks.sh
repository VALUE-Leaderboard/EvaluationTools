# execute at project root directory
cd scoring_program
echo "remember to use Python 2.7, since coco only supports 2.7"

gt_dir=../reference_data
submit_dir=../submission_data_sample
output_dir=../tmp_output
python evaluate.py ${gt_dir} ${submit_dir} ${output_dir}

cd ..
