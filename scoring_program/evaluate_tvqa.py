__author__ = "Jie Lei"

import os
import json
import numpy as np
import pprint
from os.path import join


def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Error in loading json file %s" % file_path)
        raise IOError(e.message)


def save_json_pretty(data, file_path):
    """save formatted json, use this one for some json config files"""
    with open(file_path, "w") as f:
        f.write(json.dumps(data, indent=4, sort_keys=True))


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def merge_dicts(list_dicts):
    """merge a list of dicts together"""
    result = {}
    for d in list_dicts:
        result.update(d)
    return result


def eval_tvqa_acc(predictions_path, gt_path):
    predictions = load_json(predictions_path)
    gt = load_json(gt_path)
    predictions = {int(k): int(v) for k, v in predictions.items()}
    gt = merge_dicts(list(gt["solution"].values()))
    gt = {int(k): int(v) for k, v in gt.items()}
    qids = gt.keys()

    pred_answers = []
    gt_answers = []
    for qid in qids:
        pred_answers.append(predictions[qid])
        gt_answers.append(gt[qid])

    pred_answers = np.array(pred_answers)
    gt_answers = np.array(gt_answers)
    acc = np.mean(pred_answers == gt_answers)
    return float("{:.2f}".format(100 * acc))


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", type=str, help="path to dir containing submissions")
    parser.add_argument("--gt_dir", type=str, help="path to dir containing ground-truth files")
    parser.add_argument("--output_dir", type=str, help="path to dir saving output data")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    return args


def eval_tvqa(submit_dir, truth_dir, output_dir, val_only=True):
    dataset_name = "tvqa"
    print("Evaluating task {}".format(dataset_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if not os.path.isdir(submit_dir):
        print("%s doesn't exist" % submit_dir)

    if dataset_name == "tvqa":
        output_path = join(
            output_dir, "{}_metrics.json".format(dataset_name))
        file_paths = dict(
            val=dict(
                submission=join(submit_dir, "tvqa_val_predictions.json"),
                solution=join(truth_dir, "tvqa_val_solution.json")
            ),
        )
        print("val_only", val_only)
        if not val_only:
            file_paths.update(
                test=dict(
                    submission=join(submit_dir, "tvqa_test_predictions.json"),
                    solution=join(truth_dir, "tvqa_test_solution.json")
                )
            )
    else:
        raise ValueError

    output_metrics = {}
    for split_name in file_paths:
        print("split_name ", split_name)
        output_metrics[split_name] = eval_tvqa_acc(
            file_paths[split_name]["submission"],
            file_paths[split_name]["solution"]
        )

    with open(output_path, "w") as f:
        f.write(json.dumps(output_metrics, indent=4))


if __name__ == '__main__':
    args = get_args()
    submit_dir = args.submission_dir
    truth_dir = args.gt_dir
    output_dir = args.output_dir
    eval_tvqa(submit_dir, truth_dir, output_dir)
