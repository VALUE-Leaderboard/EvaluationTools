"""
Load prediction file and GT file to calculate TVR metrics:
- recall at top K (R@K), for a specified IoU, where K in [1, 5, 10, 100], IoU in [0.5, 0.7]
"""
import os
from os.path import join
import time
import json
from evaluate_tvr import get_args, load_json, eval_retrieval, load_jsonl


def eval_yc2r(submit_dir, truth_dir, output_dir, val_only=True):
    dataset_name="yc2r"
    print("Evaluating task {}".format(dataset_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset_name == "yc2r":
        output_path = join(
            output_dir, "{}_metrics.json".format(dataset_name))
        file_paths = dict(
            val=dict(submission=join(submit_dir, "yc2r_val_predictions.json"),
                     solution=join(truth_dir, "yc2r_val_release.jsonl"),
                     output=join(output_dir, "yc2r_val_metrics.json")),
        )
        if not val_only:
            file_paths.update(
                test=dict(submission=join(submit_dir, "yc2r_test_predictions.json"),
                          solution=join(truth_dir, "yc2r_test_gt.jsonl"),
                          output=join(output_dir, "yc2r_test_metrics.json"))
            )

        video2dur_idx_path = join(truth_dir, "yc2r_video2dur_idx.json")
    else:
        raise ValueError

    start_time = time.time()
    video2dur_idx = load_json(video2dur_idx_path)
    video2idx = {split_name: {k: v[1] for k, v in data.items()}
                 for split_name, data in video2dur_idx.items()}
    output_metrics = {}
    for split_name in file_paths:
        print("Evaluating {}".format(split_name))
        submission = load_json(file_paths[split_name]["submission"])
        submission["video2idx"] = video2idx[split_name]
        gt = load_jsonl(file_paths[split_name]["solution"])
        results = eval_retrieval(submission, gt, iou_thds=(0.5, 0.7), verbose=False, use_desc_type=False)
        output_metrics[split_name] = results

    with open(output_path, "w") as f:
        f.write(json.dumps(output_metrics, indent=4))

    print("Evaluation finished in {} seconds.".format(time.time() - start_time))


if __name__ == '__main__':
    args = get_args()
    submit_dir = args.submission_dir
    truth_dir = args.gt_dir
    output_dir = args.output_dir
    eval_yc2r(submit_dir, truth_dir, output_dir)

