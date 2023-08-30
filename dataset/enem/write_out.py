# From https://github.com/piresramon/gpt-4-enem/
# Edited by viniciusarruda to get all data used in the paper
# Also, see https://github.com/piresramon/gpt-4-enem/issues/1
# This code still leads to data leakage, thus I removed it manually

import argparse
import numpy as np
import json
import os
import random
from lm_eval import tasks
from lm_eval.utils import join_iters


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_base_path", required=True)
    parser.add_argument("--tasks", default="all_tasks")
    parser.add_argument("--provide_description", action="store_true")
    parser.add_argument("--sets", type=str, default="val")  # example: val,test
    parser.add_argument("--num_fewshot", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_examples", type=int, default=1)
    parser.add_argument("--description_dict_path", default=None)
    return parser.parse_args()


def get_area(doc):
    q_id = int(doc["id"].split("_")[-1])
    area = ["languages", "human-sciences", "natural-sciences", "mathematics"][int(np.ceil(q_id / 45)) - 1]
    return area


def main():
    args = parse_args()
    np.random.seed(args.seed)

    if args.tasks == "all_tasks":
        task_names = tasks.ALL_TASKS
    else:
        task_names = args.tasks.split(",")
    task_dict = tasks.get_task_dict(task_names)

    description_dict = {}
    if args.description_dict_path:
        with open(args.description_dict_path, "r") as f:
            description_dict = json.load(f)

    os.makedirs(args.output_base_path, exist_ok=True)
    for task_name, task in task_dict.items():
        rnd = random.Random()
        rnd.seed(args.seed)

        iters = []

        for set in args.sets.split(","):
            if set == "train" and task.has_training_docs():
                docs = task.training_docs()
            if set == "val" and task.has_validation_docs():
                docs = task.validation_docs()
            if set == "test" and task.has_test_docs():
                docs = task.test_docs()
            iters.append(docs)

        docs = join_iters(iters)

        description = description_dict[task_name] if description_dict and task_name in description_dict else ""

        json_docs: list[dict] = []
        for doc in docs:
            assert "area" not in doc
            doc["area"] = get_area(doc)
            assert "prompt" not in doc
            doc["prompt"] = task.fewshot_context(
                doc=doc,
                num_fewshot=args.num_fewshot,
                rnd=rnd,
                description=description,
            )
            json_docs.append(doc)
        with open(os.path.join(args.output_base_path, f"{task_name}.json"), "w", encoding="utf-8") as f:
            json.dump(json_docs, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    main()
