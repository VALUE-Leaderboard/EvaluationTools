__author__ = "Jie Lei"

import os
import time
from os.path import join


import json
import numpy as np
import pprint


def load_json(file_path):
    try:
        with open(file_path, "r") as f:
            return json.load(f)
    except Exception as e:
        print("Error in loading json file %s" % file_path)
        raise IOError(e.message)


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


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


def eval_qa(submission, gt):
    gt_qid2ans = {int(e["example_id"]): int(e["answer"]) for e in gt}
    submission_qid2ans = {int(k): int(v) for k, v in submission.items()}
    gt_qids = set(list(gt_qid2ans.keys()))
    submission_qids = set(list(submission_qid2ans.keys()))
    assert gt_qids == submission_qids, \
        "submission example_id ids should be the same as GT example_id ids."
    qids = list(gt_qids)
    gt_array = []
    submission_array = []
    for qid in qids:
        gt_array.append(gt_qid2ans[qid])
        submission_array.append(submission_qid2ans[qid])
    gt_array = np.array(gt_array)
    submission_array = np.array(submission_array)
    acc = np.mean(gt_array == submission_array)
    acc = float("{:.2f}".format(100 * acc))
    return acc


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", type=str, help="path to dir containing submissions")
    parser.add_argument("--gt_dir", type=str, help="path to dir containing ground-truth files")
    parser.add_argument("--output_dir", type=str, help="path to dir saving output data")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    return args


def eval_violin(submit_dir, truth_dir, output_dir, val_only=True):
    dataset_name = "violin"
    print("Evaluating task {}".format(dataset_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset_name == "violin":
        output_path = join(output_dir, "violin_metrics.json")
        file_paths = dict(
            test=dict(submission=join(submit_dir, "violin_test_predictions.json"),
                      solution=join(truth_dir, "violin_test_release.jsonl")),
        )
        if not val_only:
            file_paths.update(
                test_private=dict(submission=join(submit_dir, "violin_test_private_predictions.json"),
                                  solution=join(truth_dir, "violin_test_private_gt.jsonl")),
            )
    else:
        raise ValueError

    start_time = time.time()
    output_metrics = {}
    for split_name in file_paths:
        submission = load_json(file_paths[split_name]["submission"])
        gt = load_jsonl(file_paths[split_name]["solution"])
        output_metrics[split_name] = eval_qa(submission, gt)
    save_json_pretty(output_metrics, output_path)
    print("Evaluation finished in {} seconds.".format(time.time() - start_time))


if __name__ == '__main__':
    args = get_args()
    submit_dir = args.submission_dir
    truth_dir = args.gt_dir
    output_dir = args.output_dir
    eval_violin(submit_dir, truth_dir, output_dir)

