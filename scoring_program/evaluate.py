"""
Evaluate predictions from all available tasks and then gather evaluation results.
"""
import os
import sys
import json
import time
from os.path import join
from evaluate_how2qa import eval_how2qa
from evaluate_how2r import eval_how2r
from evaluate_tvc import eval_tvc
from evaluate_tvqa import eval_tvqa
from evaluate_tvr import eval_tvr
from evaluate_vatex_en_c import eval_vatex_en_c
from evaluate_vatex_en_r import eval_vatex_en_r
from evaluate_violin import eval_violin
from evaluate_vlep import eval_vlep
from evaluate_yc2c import eval_yc2c
from evaluate_yc2r import eval_yc2r


TASK2EVAL_FUNC = dict(
    how2qa=eval_how2qa,
    how2r=eval_how2r,
    tvc=eval_tvc,
    tvr=eval_tvr,
    tvqa=eval_tvqa,
    vatex_en_r=eval_vatex_en_r,
    vatex_en_c=eval_vatex_en_c,
    violin=eval_violin,
    vlep=eval_vlep,
    yc2c=eval_yc2c,
    yc2r=eval_yc2r
)


TYPE2TASKS = dict(
    vcmr=["tvr", "how2r"],  # corpus-level moment retrieval
    vr=["yc2r", "vatex_en_r"],  # standard text-to-video retrieval
    qa=["vlep", "tvqa", "violin", "how2qa"],  # multiple choice tasks
    captioning=["yc2c", "tvc", "vatex_en_c"]  #
)

TASK2TYPE = {}
for task_type, tasks in TYPE2TASKS.items():
    for t in tasks:
        TASK2TYPE[t] = task_type

TASK2SPLIT_NAME = dict()
TASK2SPLIT_NAME["val"] = dict(
    tvr="val",
    how2r="val",
    vatex_en_r="val",
    yc2r="val",
    tvqa="val",
    vlep="dev",
    violin="test",
    how2qa="val",
    yc2c="val",
    tvc="val",
    vatex_en_c="test_public"
)
TASK2SPLIT_NAME["test"] = dict(
    tvr="test_public",
    how2r="test_public",
    vatex_en_r="test_public",
    yc2r="test",
    tvqa="test_public",
    vlep="test",
    violin="test_private",
    how2qa="test_public",
    yc2c="test",
    tvc="test_public",
    vatex_en_c="test_private"
)


def load_json(filename):
    with open(filename, "r") as f:
        return json.load(f)


def flat_list_of_lists(l):
    """flatten a list of lists [[1,2], [3,4]] to [1,2,3,4]"""
    return [item for sublist in l for item in sublist]


def gather_all_task_scores(output_root_dir, split_name="val"):
    """
    Args:
        output_root_dir: str
        split_name: str, one of ["val", "test"]

    """
    # generate a `scores.txt` file for Codalab Leaderboard.
    evaluated_tasks = list(set(get_all_subdir_names(output_root_dir)) & set(TASK2TYPE.keys()))
    print("There are {} successfully evaluated tasks in total: {}".format(len(evaluated_tasks), evaluated_tasks))

    # Retrieval: video retrieval + video corpus moment retrieval.
    # -- AveR: average of R@1 and R@10. For VCMR, use IoU=0.7
    retrieval_metrics = {}
    for task in TYPE2TASKS["vr"] + TYPE2TASKS["vcmr"]:
        if task not in evaluated_tasks:
            continue
        _metrics_path = join(output_root_dir, task, "{}_metrics.json".format(task))
        _split_name = TASK2SPLIT_NAME[split_name][task]
        _metrics = load_json(_metrics_path)[_split_name]
        _key = task
        if TASK2TYPE[task] == "vr":
            _value = (_metrics["VR"]["r1"] + _metrics["VR"]["r5"] + _metrics["VR"]["r10"]) / 3.
        else:  # vcmr
            _value = (_metrics["VCMR"]["0.7_r1"] + _metrics["VCMR"]["0.7_r5"] + _metrics["VCMR"]["0.7_r10"]) / 3.
        retrieval_metrics[_key] = float("{:.2f}".format(_value))

    if len(retrieval_metrics) > 0:
        avg = sum(list(retrieval_metrics.values())) / len(retrieval_metrics)
        retrieval_metrics["average"] = float("{:.2f}".format(avg))
    else:
        retrieval_metrics["retrieval-average"] = 0

    # QA -- acc
    qa_metrics = {}
    for task in TYPE2TASKS["qa"]:
        if task not in evaluated_tasks:
            continue
        _metrics_path = join(output_root_dir, task, "{}_metrics.json".format(task))
        _split_name = TASK2SPLIT_NAME[split_name][task]
        _metrics = load_json(_metrics_path)[_split_name]
        _key = task
        _value = _metrics
        qa_metrics[_key] = _value

    if len(qa_metrics) > 0:
        avg = sum(list(qa_metrics.values())) / len(qa_metrics)
        qa_metrics["average"] = float("{:.2f}".format(avg))
    else:
        qa_metrics["qa-average"] = 0

    # Captioning -- CIDEr
    captioning_metrics = {}
    for task in TYPE2TASKS["captioning"]:
        if task not in evaluated_tasks:
            continue
        _metrics_path = join(output_root_dir, task, "{}_metrics.json".format(task))
        _split_name = TASK2SPLIT_NAME[split_name][task]
        _metrics = load_json(_metrics_path)[_split_name]
        _key = task
        _value = _metrics["CIDEr"]
        captioning_metrics[_key] = _value

    if len(captioning_metrics) > 0:
        avg = sum(list(captioning_metrics.values())) / len(captioning_metrics)
        captioning_metrics["average"] = float("{:.2f}".format(avg))
    else:
        captioning_metrics["captioning-average"] = 0

    all_metrics = dict(
        retrieval=retrieval_metrics,
        qa=qa_metrics,
        captioning=captioning_metrics
    )
    return all_metrics


def get_all_subdir_names(root_dir_path):
    subdir_paths = os.listdir(root_dir_path)
    subdir_names = [os.path.basename(p) for p in subdir_paths]
    return subdir_names


def eval_main():
    """
    There is a fixed directory structure that the scoring program operates within. It looks like this:
    https://github.com/codalab/codalab-competitions/wiki/User_Building-a-Scoring-Program-for-a-Competition#directory-structure-for-submissions
    """
    start_time = time.time()
    is_local = True  # set to False when compose bundle
    val_only = True  # if True evaluate on `val` split only, otherwise evaluate on both `val` and `test`
    if is_local:
        gt_dir = sys.argv[1]
        submit_dir = sys.argv[2]
        output_dir = sys.argv[3]
    else:
        input_dir = sys.argv[1]  # contains user submission file and ground-truth reference file
        output_dir = sys.argv[2]  # save output files

        """
        Each task should have a separate directory named after the keys in 
        `TASK2TYPE` to contain all task-specific data. For example, for `tvr` 
        task, its user submission is at `submit_dir/tvr`, ground-truth files 
        are at `truth_dir/tvr`.
        """
        submit_dir = join(input_dir, "res")  # user submission unzipped
        gt_dir = join(input_dir, "ref")  # contains the reference data unzipped

    submitted_tasks = list(set(get_all_subdir_names(submit_dir)) & set(TASK2TYPE.keys()))
    print("There are {} submitted tasks in total: {}".format(len(submitted_tasks), submitted_tasks))

    # run evaluation in multi-process.
    for task in submitted_tasks:
        task_submission_dir = join(submit_dir, task)
        task_gt_dir = join(gt_dir, task)
        task_output_dir = join(output_dir, task)
        if not os.path.exists(task_output_dir):
            os.makedirs(task_output_dir)
        TASK2EVAL_FUNC[task](
            task_submission_dir, task_gt_dir, task_output_dir, val_only=val_only)

    # gather results
    gathered_scores = {}
    for split_name in ["val", "test"]:
        if val_only and split_name == "test":
            continue
        gathered_scores[split_name] = \
            gather_all_task_scores(output_dir, split_name=split_name)

    scores_text = []
    for split_name, split_metrics in gathered_scores.items():
        for task_group, task_group_metrics in split_metrics.items():
            for task, score in task_group_metrics.items():
                scores_text.append("{}-{}-{}: {}".format(
                    split_name, task_group, task, score))

    for split_name in gathered_scores:
        all_values = flat_list_of_lists(
            [list(e.values()) for e in gathered_scores[split_name].values()]
        )
        all_values_avg = float("{:.2f}".format(sum(all_values) / len(all_values)))
        gathered_scores[split_name]["average"] = all_values_avg
        scores_text.insert(0, "{}-average: {}".format(split_name, all_values_avg))

    gathered_scores_save_path = join(output_dir, "all_scores.json")
    with open(gathered_scores_save_path, "w") as f:
        f.write(json.dumps(gathered_scores, indent=4))

    gathered_scores_txt_save_path = join(output_dir, "scores.txt")
    with open(gathered_scores_txt_save_path, "w") as f:
        f.write("\n".join(scores_text))

    print("===> Total Evaluation finished in {} seconds.".format(time.time() - start_time))


if __name__ == '__main__':
    eval_main()
