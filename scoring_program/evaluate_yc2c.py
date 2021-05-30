import os
import json
import time
from os.path import join

from evaluate_tvc import TVRCaptionEval, get_args


def eval_yc2c(submit_dir, truth_dir, output_dir, val_only=True):
    dataset_name="yc2c"
    print("Evaluating task {}".format(dataset_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset_name == "yc2c":
        output_path = join(
            output_dir, "{}_metrics.json".format(dataset_name))
        file_paths = dict(
            val=dict(submission=join(submit_dir, "yc2c_val_predictions.jsonl"),
                     solution=join(truth_dir, "yc2c_val_release.jsonl"),
                     output=join(output_dir, "yc2c_val_metrics.json")),
        )
        if not val_only:
            file_paths.update(
                test=dict(submission=join(submit_dir, "yc2c_test_predictions.jsonl"),
                          solution=join(truth_dir, "yc2c_test_gt.jsonl"),
                          output=join(output_dir, "yc2c_test_metrics.json"))
            )
    else:
        raise ValueError

    start_time = time.time()
    output_metrics = {}
    for split_name in file_paths:
        evaluator = TVRCaptionEval(file_paths[split_name]["submission"],
                                   file_paths[split_name]["solution"])
        evaluator.evaluate()
        output_metrics[split_name] = evaluator.eval_res

    with open(output_path, "w") as f:
        f.write(json.dumps(output_metrics, indent=4))

    print("Evaluation finished in {} seconds.".format(time.time() - start_time))


if __name__ == '__main__':
    args = get_args()
    submit_dir = args.submission_dir
    truth_dir = args.gt_dir
    output_dir = args.output_dir
    eval_yc2c(submit_dir, truth_dir, output_dir)

