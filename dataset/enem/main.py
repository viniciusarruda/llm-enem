import os
import json

# Check if all few shots are the same for all

with open(os.path.join("dataset", "enem", "enem_cot_2022_3_shot.json"), "r", encoding="utf-8") as f:
    data = json.load(f)

all_few_shots = []
for d in data:
    # Check for data leakage
    if "##\nQuestão 4:\n" in d["prompt"]:
        splitted = d["prompt"].split("##\nQuestão 3:\n")
    else:
        splitted = d["prompt"].split("##\nQuestão 2:\n")
    assert len(splitted) == 2

    assert d["query"] not in splitted[0]
    # with open("data_leakage.json", "w", encoding="utf-8") as f:
    #     json.dump(d, f, indent=4, ensure_ascii=False)
    assert d["query"] in splitted[1]

    # if len(splitted) != 2:
    #     with open("test.json", "w", encoding="utf-8") as f:
    #         json.dump(d, f, indent=4, ensure_ascii=False)
    #     exit()
    # offset = len(d["query"])
    # few_shot_prompt = d["prompt"][:-offset]
    # all_few_shots.append(few_shot_prompt)


# print(len(all_few_shots))

# all_few_shots = set(all_few_shots)

# print(all_few_shots)
# with open("debug.json", "w", encoding="utf-8") as f:
#     json.dump(list(all_few_shots), f, indent=4, ensure_ascii=False)
