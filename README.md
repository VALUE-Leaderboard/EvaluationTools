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


## Retrieval Submission

Given a natural language query and a large pool of videos,
the TVR (VCMR) task requires a system to retrieve a relevant moment from the videos.
The table below shows a comparison of the TVR task and the subtasks: 

| Task | Description |
| --- | --- | 
| VCMR | *Video Corpus Moment Retrieval*. Localize a moment from a large video corpus. |
| SVMR | *Single Video Moment Retrieval*. Localize a moment from a given video. |
| VR | *Video Retrieval*. Retrieve a video from a large video corpus. |

VCMR and VR only requires a query and a video corpus, SVMR additionally requires knowing the ground-truth video. 
Thus, it is not possible to perform SVMR on our `test` set, where the ground-truth video is hidden. 


- TVR and How2R

    TVR and How2R evaluates video corpus moment retrieval (VCMR). Given a query, it requires a model to not only retrieve the most relevant video, but also the most relevant segment (or moment) inside the videos. Each prediction file for TVR or How2R should be formatted as a single `.json` file:
    ```
    {
        "VCMR": [{
                "desc_id": 90200,
                "predictions": [
                    [19614, 9.0, 12.0, 1.7275],
                    [20384, 12.0, 18.0, 1.7315],
                    [20384, 15.0, 21.0, 1.7351],
                    ...
                ]
            },
            ...
        ],
        "VR": [{
                "desc_id": 90200,
                "predictions": [19614, 20384, ...],
                    ...
                ]
            },
            ...
        ]
    }
    ```
    | entry | description |
    | --- | ----|
    | VCMR | `list(dicts)`, stores predictions for the task `VCMR`. | 
    | VR | `list(vid_id)`, stores predictions for the task `VR`. | 

    The evaluation script will evaluate the predictions for tasks `[VCMR, VR]` independently.
    Each dict in VCMR list is:
    ```
    {
        "desc": str,
        "desc_id": int,
        "predictions": [[vid_id (int), st (float), ed (float), score (float)], ...]
    }
    ```
    `predictions` is a `list` containing 100 `sublist`, each `sublist` has exactly 4 items: 
    `[vid_id (int), st (float), ed (float), score (float)]`,
    which are `vid_id` (video id), `st` and `ed` (moment start and end time, in seconds.), `score` (score of the prediction). 
    The `score` item will not be used in the evaluation script, it is left here for record. 
    
- YC2R and VATEX-EN-R
    
    For these two tasks, it is only required to return the most relevant video from a video corpus. Thus, you only need to submit the `VR` task entry described above.


## QA Submission

This task type involves 4 multiple choice QA tasks: TVQA, VIOLIN, How2QA and VLEP. Given a video with a question, the task is to select an answer from a set of candidate answers. The submissions follow the same submission format, a single `.json` file for each split:

```
{
    question_id (str): answer_id (int), 
    ...
}
```



## Caption Submission

This task type involves 3 captioning tasks: TVC, VATEX-EN-C, YC2C. Given a video (or a clip inside the video), the task is to generate a natural language description regarding the given video. The submissions follow the format, a single `.json` file for each split:

```
{
    "video_id": str, 
    "clip_id": int, 
    "descs": [{"desc": str}]
}
```

`desc` contains the generated caption sentence. `video_id` indicates a video, `clip_id` indicates a clip inside the video.  Each video can have multiple clips, thus only `clip_id` uniquely defines an example. 


## Submission Files

We have provided sample submission files in [./submission_data_sample](./submission_data_sample). Please strictly follow the submission format (including directory layout and file names) in these files. For each task, it is also required to submit both `val` and `test` predictions. 

After you have the submission files ready, please zip all the directory in a single zip file, without any extra enclosing directory. For example, you can use the following command to zip the sample predictions provided in this repo:
```
cd submission_data_sample && zip -r submission_sample.zip ./*
```
Next, you can submit this `zip` file to our CodaLab evaluation portal.
