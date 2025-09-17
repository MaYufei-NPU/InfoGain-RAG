import json
import pandas as pd
from ..utils import write_jsonl


def read_triviaqa_train(path: str, save_path: str):

    new_data = []
    with open(path, "r") as f:
        # data = json.loads(f)
        data = f.read()
        data = json.loads(data)

        for d in data["Data"]:

            new_data.append(
                {
                    "raw_idx": d["QuestionId"],
                    "query": d["Question"],
                    "answers": [d["Answer"]["Value"]],
                }
            )

    write_jsonl(new_data, save_path)


def read_triviaqa_dev(path: str, save_path: str):

    new_data = []
    with open(path, "r") as f:
        # data = json.loads(f)
        data = f.read()
        data = json.loads(data)

        for d in data["Data"]:

            new_data.append(
                {
                    "raw_idx": d["QuestionId"],
                    "query": d["Question"],
                    "answers": [d["Answer"]["Value"]],
                }
            )

    write_jsonl(new_data, save_path)


def merge_passage_train(
    passage_path: str, raw_path: str, save_path: str, left: int = 0, right: int = 0
):

    with open(passage_path, "r") as f:
        passages_raw_data = json.load(f)

    passage_data = []
    for d in passages_raw_data:
        for i, q in enumerate(d["top_passages"]):
            # if i > 4:
            #     break
            if i < left or i >= right:
                continue

            doc_pair = {"query": d["question"]}
            doc_pair["passage_id"] = q["id"]
            doc_pair["passage_title"] = q["title"]
            doc_pair["passage_text"] = q["text"]
            doc_pair["passage_rank"] = i

            passage_data.append(doc_pair)

    with open(raw_path, "r") as f:
        raw_data = {}
        for line in f:
            d = json.loads(line)
            raw_data[d["query"]] = d

    new_data = []

    for pdz in passage_data:
        if pdz["query"] in raw_data:
            new_data.append(
                {
                    "query": pdz["query"],
                    "passage_id": pdz["passage_id"],
                    "passage_title": pdz["passage_title"],
                    "passage_text": pdz["passage_text"],
                    "passage_rank": pdz["passage_rank"],
                    "answers": raw_data[pdz["query"]]["answers"],
                    "raw_idx": raw_data[pdz["query"]]["raw_idx"],
                }
            )

    final_data = []
    for d in new_data:
        for a in d["answers"]:
            final_data.append(
                {
                    "query": d["query"],
                    "passage_id": d["passage_id"],
                    "passage_title": d["passage_title"],
                    "passage_text": d["passage_text"],
                    "passage_rank": d["passage_rank"],
                    "answer": a,
                    "raw_idx": d["raw_idx"],
                    "dataset_name": "trivia_qa",
                    "dataset_type": "train",
                }
            )

    df = pd.DataFrame(final_data)
    df.to_parquet(save_path)
