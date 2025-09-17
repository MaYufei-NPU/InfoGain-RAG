import gzip
import json
import random
import os
import jsonlines
from multiprocessing import Pool, cpu_count
import pandas as pd
from ..utils import read_gz_jsonl, read_jsonl, random_sample, write_jsonl


def sample_naturalqa_data(file_dir: str, save_path: str):
    files = os.listdir(file_dir)
    datas = []
    for file in files:
        print(file)
        file_path = os.path.join(file_dir, file)
        data = []
        with gzip.open(file_path, "rt", encoding="utf-8") as f:
            with jsonlines.Reader(f) as reader:
                for line in f:
                    line = json.loads(line)
                    print(line["annotations"][0]["short_answers"])
                    if line["annotations"][0]["short_answers"]:
                        data.append(line)
        datas.extend(random.sample(data, 200))
    # datas = random_sample(datas, 10000)
    write_jsonl(datas, save_path)


def extract_short_answers(document_tokens, short_answers):
    extracted_texts = []

    for answer in short_answers:
        start_idx = answer["start_token"]
        end_idx = answer["end_token"]

        text_parts = []

        for i in range(start_idx, end_idx):
            token_info = document_tokens[i]
            if not token_info["html_token"]:  # 排除 html token
                text_parts.append(token_info["token"])

        # 连接有效的文本部分
        extracted_text = " ".join(text_parts)
        extracted_texts.append(extracted_text)

    return extracted_texts


def preprocess_token_from_document(sample_path: str, save_path: str):
    final_data = []
    with jsonlines.open(sample_path) as reader:
        for line in reader:
            raw_idx = line["example_id"]
            print(raw_idx)
            query = line["question_text"]
            answers = extract_short_answers(
                line["document_tokens"], line["annotations"][0]["short_answers"]
            )
            final_data.append({"raw_idx": raw_idx, "query": query, "answers": answers})
    write_jsonl(final_data, save_path)


def process_train_file(file: str, file_dir: str):
    file_path = os.path.join(file_dir, file)
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        with jsonlines.Reader(f) as reader:
            for line in f:
                line = json.loads(line)
                # print(line["annotations"][0]["short_answers"])
                if line["annotations"][0]["short_answers"]:
                    data.append(line)
    final_data = []
    for line in data:
        raw_idx = line["example_id"]
        query = line["question_text"]
        answers = extract_short_answers(
            line["document_tokens"], line["annotations"][0]["short_answers"]
        )
        final_data.append({"raw_idx": raw_idx, "query": query, "answers": answers})

    return final_data


def preprocess_token_from_dataset_train(file_dir: str, save_path: str):
    files = os.listdir(file_dir)

    # Create a pool of worker processes
    pool = Pool(cpu_count())

    total_final_data = []

    # Map the process_file function to the list of files, running in parallel
    for final_data in pool.imap_unordered(process_train_file, files):
        total_final_data.extend(final_data)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()

    write_jsonl(total_final_data, save_path)


def process_dev_file(file: str, file_dir: str):
    file_path = os.path.join(file_dir, file)
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        with jsonlines.Reader(f) as reader:
            for line in f:
                line = json.loads(line)
                if line["annotations"][0]["short_answers"]:
                    data.append(line)
    final_data = []
    for line in data:
        raw_idx = line["example_id"]
        query = line["question_text"]
        answers = extract_short_answers(
            line["document_tokens"], line["annotations"][0]["short_answers"]
        )
        final_data.append({"raw_idx": raw_idx, "query": query, "answers": answers})
    return final_data


def preprocess_token_from_dataset_dev(file_dir: str, save_path: str):
    files = os.listdir(file_dir)

    # Create a pool of worker processes
    pool = Pool(cpu_count())

    total_final_data = []

    # Map the process_file function to the list of files, running in parallel
    for final_data in pool.imap_unordered(process_dev_file, files):
        total_final_data.extend(final_data)

    # Close the pool and wait for the work to finish
    pool.close()
    pool.join()


def merge_passage_train(
    left: int, right: int, passage_path: str, raw_path: str, save_path: str
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
                    "dataset_name": "nq",
                    "dataset_type": "train",
                }
            )

    df = pd.DataFrame(final_data)
    df.to_parquet(save_path)
