# VALUE Benchmark Evaluation Tools 

This repository hosts evaluation tools for the [VALUE benchmark](https://value-leaderboard.github.io), including evaluation code and sample submissions.


## Evaluation Code

This code requires Python 2.7 (for captioning evaluation) and NumPy.

0. Clone this repository.
    ```
    git clone git@github.com:VALUE-Leaderboard/EvaluationTools.git
    ```

1. Run evaluation on sample predictions at [./submission_data_sample](./submission_data_sample)
    ```
    bash scripts/scripts/run_local_all_tasks.sh
    ```
    This evaluates all the task predictions for their respective validation split. Note that the test annotations are reserved, you have to submit to our CodaLab leaderboard for evaluation. The output will be written into `tmp_output` To evaluate only a single task, please run:
    ```
    bash scripts/run_local_all_tasks.sh
    ```


## Submission Files
We have provided sample submission files in [./submission_data_sample](./submission_data_sample). Please strictly follow the submission format (including directory layout and file names) in these files. For each task, it is also required to submit both `val` and `test` predictions. 

After you have the submission files ready, please zip all the directory in a single zip file, without any extra enclosing directory. For example, you can use the following command to zip the sample predictions provided in this repo:
```
cd submission_data_sample && zip -r submission_sample.zip ./*
```
Next, you can submit this `zip` file to our CodaLab evaluation portal.
