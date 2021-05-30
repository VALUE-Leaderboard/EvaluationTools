# execute at project root directory
task_name=$1  # [tvr, tvc]
echo "remember to use Python 2.7, since coco only supports 2.7"
echo "Task name ${task_name}"
cd scoring_program

gt_dir=../reference_data/${task_name}
submit_dir=../submission_data_sample/${task_name}
output_dir=../tmp_output/${task_name}
python evaluate_${task_name}.py \
--gt_dir ${gt_dir} \
--submission_dir ${submit_dir} \
--output_dir ${output_dir} 

cd ..

