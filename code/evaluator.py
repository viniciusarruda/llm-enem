import os
import re
import json
import fire
from chat_completion_wrapper import (
    OpenAIChatCompletionWrapper,
    HFLlama2ChatCompletionWrapper,
    HFFalconChatCompletionWrapper,
)
from tqdm import tqdm

AREA_MAP = {
    "languages": "Languages and Codes",
    "human-sciences": "Human Sciences",
    "natural-sciences": "Natural Sciences",
    "mathematics": "Mathematics",
}

DATASET_TO_FILENAME = {
    "Zero-shot": "enem_2022_0_shot.json",
    "Few-shot": "enem_2022_3_shot.json",
    "Few-shot with Chain-of-Thought": "enem_cot_2022_3_shot.json",
}

DATASET_TO_TITLE = {
    "Zero-shot": "zero-shot",
    "Few-shot": "three-shot",
    "Few-shot with Chain-of-Thought": "three-shot<br>with CoT",
}


def get_llm(model):
    if model.startswith("gpt-3.5-turbo") or model.startswith("gpt-4"):
        assert "OPENAI_API_KEY" in os.environ, "You need to set OPENAI_API_KEY in your environment variable."
        llm = OpenAIChatCompletionWrapper(model=model, log=False)
    elif model.startswith("LLaMA-2"):
        llm = HFLlama2ChatCompletionWrapper(
            endpoint_url=os.environ[f"huggingface_{model.replace('-', '')}_url"],
            token=os.environ["huggingface_token"],
            namespace=os.environ["huggingface_namespace"],
            name=os.environ[f"huggingface_{model.replace('-', '')}_name"],
            log=False,
        )
    elif model.startswith("Falcon"):
        llm = HFFalconChatCompletionWrapper(
            endpoint_url=os.environ[f"huggingface_{model.replace('-', '')}_url"],
            token=os.environ["huggingface_token"],
            namespace=os.environ["huggingface_namespace"],
            name=os.environ[f"huggingface_{model.replace('-', '')}_name"],
            log=False,
        )
    else:
        raise ValueError(f"Model {model} is not available.")

    return llm


def format_enem_dataset(dataset: list[dict]) -> dict:
    """return the list of dicts into a data dict splitting by area and then by id"""

    new_dataset = {}
    for d in dataset:
        area = AREA_MAP[d["area"]]
        if area not in new_dataset:
            new_dataset[area] = {}
        assert d["id"] not in new_dataset[area]
        new_dataset[area][d["id"]] = d

    return new_dataset


def get_formated_answer(question, answer):
    gold = ["A.", "B.", "C.", "D.", "E."][question["gold"]]
    pred = answer

    # regex processing. Useful for zero-shot
    match_1 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDE])\.", pred)
    match_2 = re.findall(r"(?:|[Ll]etra |[Aa]lternativa )([ABCDE])", pred)
    if len(match_1) > 0:
        pred = match_1[-1] + "."
    elif len(match_2) > 0:
        pred = match_2[-1] + "."
    else:
        print(f"Regex failed at processing {pred=}")
        print(f"{gold=}, {pred=}")

    return pred, gold


def get_dataset(dataset_name):
    filename = DATASET_TO_FILENAME[dataset_name]
    with open(os.path.join("..", "dataset", "enem", filename), "r", encoding="utf-8") as f:
        data = json.load(f)
    return format_enem_dataset(data)


def evaluate(
    models: list[str] = ["gpt-3.5-turbo-0613", "gpt-4-0613"],
    dataset_names: list[str] = ["Zero-shot", "Few-shot", "Few-shot with Chain-of-Thought"],
):
    """Evaluates LLMs on Enem

    Args:
        models (list[str]): List of LLMs. Defaults to "['gpt-3.5-turbo-0613', 'gpt-4-0613']".
        dataset_names (list[str]): List of dataset names. Defaults to "['Zero-shot', 'Few-shot', 'Few-shot with Chain-of-Thought']".
    """
    # getting anwers
    for model in tqdm(models, desc="Evaluating all"):
        llm = get_llm(model)
        for dataset_name in tqdm(dataset_names, desc=f"Evaluating model {model}", leave=False):
            dataset = get_dataset(dataset_name)
            report = []
            for area, questions_by_area in dataset.items():
                for question in tqdm(questions_by_area.values(), desc=f"Evaluating area {area}", leave=False):
                    assert area == AREA_MAP[question["area"]]
                    llm.new_session()
                    answer = llm(question["prompt"])
                    pred, gold = get_formated_answer(question, answer)
                    report.append(
                        dict(
                            id=question["id"],
                            response=answer,
                            pred=pred,
                            gold=gold,
                            area=question["area"],
                        )
                    )
                    # break  # only to test, remove after testing
            with open(
                os.path.join("..", "reports", f"{model}_{DATASET_TO_FILENAME[dataset_name]}"), "w", encoding="utf-8"
            ) as f:
                json.dump(report, f, indent=4, ensure_ascii=False)


def build_results_table(
    models: list[str] = ["gpt-3.5-turbo-0613", "gpt-4-0613"],
    dataset_names: list[str] = ["Zero-shot", "Few-shot", "Few-shot with Chain-of-Thought"],
):
    """Build the results table from the evaluation reports

    Args:
        models (list[str]): List of LLMs. Defaults to "['gpt-3.5-turbo-0613', 'gpt-4-0613']".
        dataset_names (list[str]): List of dataset names. Defaults to "['Zero-shot', 'Few-shot', 'Few-shot with Chain-of-Thought']".
    """
    # getting acc
    table = {}
    for model in models:
        table[model] = {}
        for dataset_name in dataset_names:
            table[model][dataset_name] = {
                "is_correct": {v: 0 for v in list(AREA_MAP.values()) + ["Total"]},
                "count": {v: 0 for v in list(AREA_MAP.values()) + ["Total"]},
            }

            with open(
                os.path.join("..", "reports", f"{model}_{DATASET_TO_FILENAME[dataset_name]}"), "r", encoding="utf-8"
            ) as f:
                report = json.load(f)

            for item in report:
                is_correct = item["pred"] == item["gold"]
                table[model][dataset_name]["is_correct"][AREA_MAP[item["area"]]] += is_correct
                table[model][dataset_name]["is_correct"]["Total"] += is_correct
                table[model][dataset_name]["count"][AREA_MAP[item["area"]]] += 1
                table[model][dataset_name]["count"]["Total"] += 1

    # formatting as html table
    html_table = '<table border="1px">\n'

    html_table += "\t<tr>\n"
    html_table += f"\t\t<th rowspan=2>Area</th>\n"
    for model in models:
        html_table += f"\t\t<th colspan={len(dataset_names)}>{model}</th>\n"
    html_table += "\t</tr>\n"

    html_table += "\t<tr>\n"
    for model in models:
        for dataset_name in dataset_names:
            html_table += f"\t\t<th>{DATASET_TO_TITLE[dataset_name]}</th>\n"
    html_table += "\t</tr>\n"

    for area in list(AREA_MAP.values()) + ["Total"]:
        html_table += "\t<tr>\n"
        html_table += f"\t\t<td>{area}</td>\n"
        for model in models:
            for dataset_name in dataset_names:
                is_correct = table[model][dataset_name]["is_correct"][area]
                count = table[model][dataset_name]["count"][area]
                acc = is_correct / count
                html_table += f"\t\t<td>{is_correct}/{count} ({acc:.2%})</td>\n"
        html_table += "\t</tr>\n"

    html_table += "</table>"

    with open(os.path.join("..", "reports", "results.html"), "w", encoding="utf-8") as f:
        f.write(html_table)


if __name__ == "__main__":
    fire.Fire(
        {
            "evaluate": evaluate,
            "build_results_table": build_results_table,
        }
    )
    # models = ["gpt-3.5-turbo-0613", "gpt-4-0613", "Falcon-7B", "LLaMA-2-7B"]
    # dataset_names = ["Zero-shot", "Few-shot", "Few-shot with Chain-of-Thought"]
    # evaluate(models, dataset_names)
    # build_results_table(models, dataset_names)
