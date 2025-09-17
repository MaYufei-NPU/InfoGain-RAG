from typing import TypedDict, Union, List, Any
import json
import os
import pandas as pd
import uuid
from typing import List, Any, Dict


class QueryData(TypedDict):
    id: str
    # custom_id: str
    query: str
    answers: Union[str, List[Any]]
    model: str


def get_raw_popqa_data(save_dir: str) -> List[Dict[str, Any]]:
    save_path = os.path.join(save_dir, "popQA.tsv")
    if not os.path.exists(save_path):
        raise FileNotFoundError("popQA.tsv not found.")

    popqa_data = pd.read_csv(save_path, sep="\t")
    popqa_data["possible_answers"].apply(list)
    return popqa_data.to_dict(orient="records")


def get_query_dataset(raw_data: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        QueryData(
            id=d["id"],
            # custom_id="popqa-" + str(uuid.uuid4()),
            query=d["question"],
            answers=d["possible_answers"],
            model="raw",
        )
        for d in raw_data
    ]
