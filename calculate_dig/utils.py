import jsonlines
import gzip
import json
import random


def read_gz_jsonl(file_path):
    data = []
    with gzip.open(file_path, "rt", encoding="utf-8") as f:
        with jsonlines.Reader(f) as reader:
            for line in f:
                data.append(json.loads(line))
    return data


def read_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def random_sample(datas, num_samples):
    per_num_sample = (
        num_samples // len(datas)
        if num_samples % len(datas) == 0
        else num_samples // len(datas) + 1
    )
    samples = []
    for data in datas:
        samples.extend(random.sample(data, per_num_sample))
    return samples


def write_jsonl(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
