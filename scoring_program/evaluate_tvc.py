import os
import sys
import json
import time
from os.path import join

sys.path.insert(0, "./scoring_program")
sys.path.insert(0, "./pycocoevalcap")
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


def load_jsonl(filename):
    with open(filename, "r") as f:
        return [json.loads(l.strip("\n")) for l in f.readlines()]


def save_json(data, filename, save_pretty=False, sort_keys=False):
    with open(filename, "w") as f:
        if save_pretty:
            f.write(json.dumps(data, indent=4, sort_keys=sort_keys))
        else:
            json.dump(data, f)


class TVRCaptionEval:
    """
    ground_truth_path: str, .jsonl file path to the ground truth captions
        Example line in the ground_truth_path file:
        {
             "vid_name": "friends_s08e08_seg01_clip_00",
             "duration": 61.03,
             "ts": [5.8, 8.24],
             "clip_id": 86618,
             "descs": [
                 {"desc": "Rachael walks up to Phoebe and Phoebe turns around.",
                  "type": "v",
                  "from_retrieval": false,
                  "desc_id": 109026
                  },
                  ...  # (ground-truth will have 4 such entries)
             ]
        }
    prediction_path: str, .jsonl file path to the generated captions
        Example line in the ground_truth_path file: (same structure as ground_truth but many entries are missing)
        {
             "clip_id": 86618,
             "descs": [
                {"desc": "Rachael walks up to Phoebe and Phoebe turns around."}
             ]  # if multiple descriptions are given, only use the first one in the list.
        }
    """

    def __init__(self, prediction_path, ground_truth_path):
        self.ground_truth = self.load_captions(ground_truth_path, is_ground_truth=True)
        self.prediction = self.load_captions(prediction_path, is_ground_truth=False)
        self.eval_res = {}
        self.eval_res_by_clip = {}  # TODO add eval res by clip

    @classmethod
    def load_captions(cls, filename, is_ground_truth=False):
        captions = load_jsonl(filename)
        if is_ground_truth:
            return {c["clip_id"]: [{"caption": remove_nonascii(e["desc"])} for e in c["descs"]] for c in captions}
        else:
            return {c["clip_id"]: [{"caption": remove_nonascii(c["descs"][0]["desc"])}] for c in captions}

    def evaluate(self):
        # =================================================
        # Tokenization
        # =================================================
        print("Tokenization")
        tokenizer = PTBTokenizer()
        gts = tokenizer.tokenize(self.ground_truth)
        preds = tokenizer.tokenize(self.prediction)

        # =================================================
        # Setup scorers
        # =================================================
        print("Setting up scorers...")
        scorers = [
            (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
            (Meteor(), "METEOR"),
            (Rouge(), "ROUGE_L"),
            (Cider(), "CIDEr"),
            # (Spice(), "SPICE")
        ]

        # =================================================
        # Compute scores
        # =================================================
        for scorer, method in scorers:
            print("Computing {} score...".format(scorer.method()))
            score, scores = scorer.compute_score(gts, preds)
            if isinstance(method, list):
                for sc, scs, m in zip(score, scores, method):
                    self.eval_res[m] = float("{:.2f}".format(sc * 100))
            else:
                self.eval_res[method] = float("{:.2f}".format(score * 100))


def get_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission_dir", type=str, help="path to dir containing submissions")
    parser.add_argument("--gt_dir", type=str, help="path to dir containing ground-truth files")
    parser.add_argument("--output_dir", type=str, help="path to dir saving output data")
    args = parser.parse_args()
    return args


def eval_tvc(submit_dir, truth_dir, output_dir, val_only=True):
    dataset_name = "tvc"
    print("Evaluating task {}".format(dataset_name))

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if dataset_name == "tvc":
        output_path = join(
            output_dir, "{}_metrics.json".format(dataset_name))
        file_paths = dict(
            val=dict(submission=join(submit_dir, "tvc_val_predictions.jsonl"),
                     solution=join(truth_dir, "tvc_val_archive.jsonl"),
                     output=join(output_dir, "tvc_val_metrics.json")),
        )
        if not val_only:
            file_paths.update(
                test_public=dict(submission=join(submit_dir, "tvc_test_public_predictions.jsonl"),
                                 solution=join(truth_dir, "tvc_test_public_archive.jsonl"),
                                 output=join(output_dir, "tvc_test_public_metrics.json"))
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
    eval_tvc(submit_dir, truth_dir, output_dir)
