import os
import json
import time
from os.path import join

from evaluate_tvc import TVRCaptionEval, get_args


def eval_vatex_en_c(submit_dir, truth_dir, output_dir, val_only=True):
    dataset_name="vatex_en_c"
    print("Evaluating task {}".format(dataset_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset_name == "vatex_en_c":
        output_path = join(
            output_dir, "{}_metrics.json".format(dataset_name))
        file_paths = dict(
            test_public=dict(submission=join(submit_dir, "vatex_en_c_test_public_predictions.jsonl"),
                             solution=join(truth_dir, "vatex_en_c_test_public_release.jsonl"),
                             output=join(output_dir, "vatex_en_c_test_public_metrics.json")),
        )
        if not val_only:
            file_paths.update(
                test_private=dict(submission=join(submit_dir, "vatex_en_c_test_private_predictions.jsonl"),
                                  solution=join(truth_dir, "vatex_en_c_test_private_gt.jsonl"),
                                  output=join(output_dir, "vatex_en_c_test_private_metrics.json"))
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
    eval_vatex_en_c(submit_dir, truth_dir, output_dir)
