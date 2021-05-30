from os.path import join
import os
import json
import pprint
import numpy as np


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def get_rounded_percentage(float_number, n_floats=2):
    return round(float_number * 100, n_floats)


def divide_with_zero(numerator, divisor):
    return 0 if divisor == 0 else 1.0 * numerator / divisor


def get_acc_1d_bool_array(array_1d, rounded_percentage=True):
    """
    Args:
        array_1d: np.array of bool values,
        rounded_percentage: bool, if True use get_rounded_percentage else do not use it

    Returns:

    """
    acc = divide_with_zero(np.sum(array_1d), len(array_1d))
    return get_rounded_percentage(acc) if rounded_percentage else acc


def eval_qa_acc(gt_ans, pred_ans):
    eval_res = {}
    gt_ans = np.array(gt_ans)
    pred_ans = np.array(pred_ans)

    # overall QA accuracy
    eval_res["qa_acc"] = dict(
        overall=get_acc_1d_bool_array(gt_ans == pred_ans, rounded_percentage=True),
    )
    return eval_res


def eval_acc_from_files(gt_path, submission_path, skip_missing=False):
    # load + preprocess data
    gt_data = load_jsonl(gt_path)
    submission_data = load_json(submission_path)
    return eval_acc_from_data(gt_data, submission_data, skip_missing=skip_missing)


def eval_acc_from_data(gt_data, submission_data, skip_missing=False):
    print("Loaded {} GT lines, {} submission lines".format(len(gt_data), len(submission_data)))

    gt_id2ans = {int(d["example_id"]): int(d["answer"]) for d in gt_data}
    pred_id2ans = {int(k): int(v) for k, v in submission_data.items()}
    gt_ids = list(gt_id2ans.keys())

    pred_ans = []
    gt_ans = []
    skipped = []
    for k in gt_ids:
        if k in pred_id2ans:
            pred_ans.append(pred_id2ans[k])
            gt_ans.append(gt_id2ans[k])
        else:
            if skip_missing:
                skipped.append(k)
                continue
            else:
                raise ValueError("one id {} from ground-truth file is missing from your predictions.".format(k))

    Warning("\n\nYou have skipped {} examples from the ground-truth file, "
            "i.e., Your predictions do not contain these examples. "
            "Example skipped ids: {}\n\n".format(len(skipped), skipped[:3]))
    print("Evaluating {} examples, missing {}"
          .format(len(pred_ans), len(gt_data) - len(pred_ans)))
    # eval + print + save
    results = eval_qa_acc(gt_ans, pred_ans)
    if len(skipped) > 0:
        results["skipped_samples"] = \
            "Your predistions missing {} examples: e.g. (only shown 3 here), {}".format(len(skipped), skipped[:3])
    return results


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", type=str, help="path to dir containing submissions")
    parser.add_argument("--gt_dir", type=str, help="path to dir containing ground-truth files")
    parser.add_argument("--output_dir", type=str, help="path to dir saving output data")
    args = parser.parse_args()
    pprint.pprint(vars(args))
    return args


def eval_vlep(submit_dir, truth_dir, output_dir, val_only=True):
    dataset_name = "vlep"
    print("Evaluating task {}".format(dataset_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_path = join(
        output_dir, "{}_metrics.json".format(dataset_name))
    file_paths = dict(
        dev=dict(submission=join(submit_dir, "vlep_dev_predictions.json"),
                 solution=join(truth_dir, "vlep_dev_archive.jsonl"),
                 output=join(output_dir, "vlep_dev_metrics.json")),
    )
    if not val_only:
        file_paths.update(
            test=dict(submission=join(submit_dir, "vlep_test_predictions.json"),
                      solution=join(truth_dir, "vlep_test_archive.jsonl"),
                      output=join(output_dir, "vlep_test_metrics.json"))
        )

    output_metrics = {}
    for split_name in file_paths:
        results = eval_acc_from_files(
            gt_path=file_paths[split_name]["solution"],
            submission_path=file_paths[split_name]["submission"],
            skip_missing=False)
        output_metrics[split_name] = results["qa_acc"]["overall"]

    with open(output_path, "w") as f:
        f.write(json.dumps(output_metrics, indent=4))


if __name__ == '__main__':
    args = get_args()
    submit_dir = args.submission_dir
    truth_dir = args.gt_dir
    output_dir = args.output_dir
    eval_vlep(submit_dir, truth_dir, output_dir)

