import pandas as pd
import json
import os


def json_to_parquet(json_file, parquet_file):
    """
    Converts a json file to a parquet file.
    """
    with open(json_file, "r") as f:
        data = json.load(f)
    new_data = []
    for query_doc_pair in data:

        for i, doc in enumerate(query_doc_pair["top_passages"]):
            doc_pair = {
                "query": query_doc_pair["question"],
                "answers": query_doc_pair["answers"],
            }
            doc_pair["passage_id"] = doc["id"]
            print(doc["id"])
            doc_pair["passage_title"] = doc["title"]
            doc_pair["passage_text"] = doc["text"]
            doc_pair["passage_rank"] = i

            new_data.append(doc_pair)

    df = pd.DataFrame(new_data)
    df.to_parquet(parquet_file, index=False)


def convert_file(raw_data_dir: str, parquet_data_dir: str):
    raw_file_list = [
        "nq_test_top_passages.json",
        "nq_train_top_passages.json",
        "pop_test_top_passages.json",
    ]

    for raw_file in raw_file_list:
        json_file = os.path.join(raw_data_dir, raw_file)
        if not os.path.exists(parquet_data_dir):
            os.makedirs(parquet_data_dir)
        parquet_file = os.path.join(
            parquet_data_dir, raw_file.replace(".json", ".parquet")
        )
        json_to_parquet(json_file, parquet_file)


def read_parquet(parquet_file):
    return pd.read_parquet(parquet_file)
