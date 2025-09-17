from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from typing import List, Dict, Any
import pandas as pd
import uuid
import copy
from multiprocessing import Pool, cpu_count


model_path = "Qwen/Qwen2.5-7B-Instruct"


shot = """
    ## passages:

    PASSAGE1:title: 1950 in the United States: October–December\n text:October 2 &ndash; The comic strip Peanuts by Charles M. Schulz is first published in seven U.S. newspapers.,

    PASSAGE2:title:the president of the United States\n text: In 1950, the president of the USA is Harry Truman.,

    ## instruction:

    Based on the given passages, answer the given question. The answer should be as short as possible. Please ensure your answers are based solely on the information provided in the passages and do not rely on your personal knowledge. If the answer is not contained within the passages, output "Unable to answer definitively".

    ## question:
    
    Who was President when the first Peanuts cartoon was published?

    ## answer:
    """.strip()


rag_messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": shot},
    {"role": "assistant", "content": "Harry Truman"},
]


message_temp = """
    ## passages:

    PASSAGE1:title: {passage_title}\n text: {passage_text}

    ## instruction:

    Based on the given passages, answer the given question. The answer should be as short as possible. Please ensure your answers are based solely on the information provided in the passages and do not rely on your personal knowledge. If the answer is not contained within the passages, output "Unable to answer definitively".

    ## question:
    
    {query}

    ## answer:
    """.strip()


non_shot = """instruction:\n Answer the given question.\n The answer should be as short as possible.\n question:Who was President when the first Peanuts cartoon was published?\n answer:"""


NON_RAG_PROMPT = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": non_shot},
    {"role": "assistant", "content": "Harry Truman"},
]

non_shot_temp = """instruction:\n Answer the given question.\n The answer should be as short as possible.\n question:{query}\n answer:"""


def init_llm():
    # tokenizer = AutoTokenizer.from_pretrained(model_path)
    llm = LLM(
        model=model_path,
        # tensor_parallel_size=8,
        tensor_parallel_size=4,
        gpu_memory_utilization=0.5,
        dtype="half",
        # max_num_seqs=50,
        max_model_len=2048,
        enable_chunked_prefill=False,
        # max_logprobs=tokenizer.vocab_size
    )
    tokenizer = llm.get_tokenizer()
    return llm, tokenizer


def load_dataset(file_paths: List[str]):
    total_data = []
    for fp in file_paths:
        data = pd.read_parquet(fp)
        # print(data.head(2))
        data = data.to_dict("records")
        total_data.extend(data)
    # print(total_data[0])
    for d in total_data:
        d["idx"] = str(uuid.uuid4())

    return total_data


llm, tokenizer = init_llm()


def process_single_data(args):
    d = args

    new_d = copy.deepcopy(d)
    rag_prompt = rag_messages + [
        {
            "role": "user",
            "content": message_temp.format(
                passage_text=d["passage_text"],
                passage_title=d["passage_title"],
                query=d["query"],
            ),
        },
        {"role": "assistant", "content": d["answer"]},
    ]
    non_rag_prompt = NON_RAG_PROMPT + [
        {
            "role": "user",
            "content": non_shot_temp.format(
                query=d["query"],
            ),
        },
        {"role": "assistant", "content": d["answer"]},
    ]
    rtokens = tokenizer.apply_chat_template(
        rag_prompt, tokenize=True, add_generation_prompt=False
    )
    key_rtokens = tokenizer.tokenize(rag_prompt[-1]["content"])
    rag_right_idx = len(rtokens) - 2
    rag_left_idx = rag_right_idx - len(key_rtokens)

    ntokens = tokenizer.apply_chat_template(
        non_rag_prompt, tokenize=True, add_generation_prompt=False
    )
    nrag_right_idx = len(ntokens) - 2
    nrag_left_idx = nrag_right_idx - len(key_rtokens)

    new_d["non_rag_prompt"] = non_rag_prompt
    new_d["rag_prompt"] = rag_prompt
    new_d["rag_left_idx"] = rag_left_idx
    new_d["rag_right_idx"] = rag_right_idx
    new_d["non_rag_left_idx"] = nrag_left_idx
    new_d["non_rag_right_idx"] = nrag_right_idx

    return new_d


def prepare_prompts(data: List[Dict[str, Any]]):
    final = []
    args = [d for d in data]

    with Pool(cpu_count()) as pool:
        final = pool.map(process_single_data, args)

    return final


def kmp_search(A, B):
    # 计算B的部分匹配表
    def compute_lps(pattern):
        lps = [0] * len(pattern)
        length = 0  # 长度最长的前缀后缀
        i = 1
        while i < len(pattern):
            if pattern[i] == pattern[length]:
                length += 1
                lps[i] = length
                i += 1
            else:
                if length != 0:
                    length = lps[length - 1]
                else:
                    lps[i] = 0
                    i += 1
        return lps

    lps = compute_lps(B)  # 获取B的部分匹配表
    positions = []  # 存储匹配的位置
    i = 0  # A的指针
    j = 0  # B的指针

    while i < len(A):
        if A[i] == B[j]:
            i += 1
            j += 1

        if j == len(B):  # 找到一个匹配
            positions.append(i - j)
            j = lps[j - 1]
        elif i < len(A) and A[i] != B[j]:  # 字符不匹配
            if j != 0:
                j = lps[j - 1]
            else:
                i += 1
    return positions


def find_all_sublist_positions(A, B):
    len_A = len(A)
    len_B = len(B)
    positions = []

    # 遍历 A，尝试从每个可能的起始位置找到 B
    for i in range(len_A - len_B + 1):  # 确保不会越界
        if A[i : i + len_B] == B:  # 比较 A 的切片与 B
            positions.append(i)  # 如果找到，记录下起始索引

    return positions  # 返回所有找到的起始索引


def find_last_occurrence(A, B):
    len_A = len(A)
    len_B = len(B)

    last_index = -1  # 记录最后一次出现的位置

    # 遍历A，检查B是否在其子集中
    for i in range(len_A - len_B + 1):
        if A[i : i + len_B] == B:
            last_index = i  # 更新最后出现的位置

    return last_index


def parse_prob(txt, is_rag, left_idx, right_idx):
    prompt_logprobs = txt.prompt_logprobs
    logprobs = [
        list(d.values())[0].logprob
        for i, d in enumerate(prompt_logprobs)
        if i in range(left_idx, right_idx)
    ]

    return logprobs


def read_parquet(file_path):
    df = pd.read_parquet(file_path)
    return df


def per_loop(cur_idx, step_len, data_with_prompt):
    rag_prompts = [d["rag_prompt"] for d in data_with_prompt]
    # print(rag_prompts[0])
    rag_text = tokenizer.apply_chat_template(
        rag_prompts, tokenize=False, add_generation_prompt=False
    )

    rag_resp = llm.generate(
        rag_text,
        sampling_params=SamplingParams(
            temperature=0.8, top_p=0.85, max_tokens=1, prompt_logprobs=0
        ),
    )
    cnt = 0
    for rag_txt in rag_resp:
        left_idx = data_with_prompt[cnt]["rag_left_idx"]
        right_idx = data_with_prompt[cnt]["rag_right_idx"]
        prob = parse_prob(rag_txt, True, left_idx, right_idx)
        data_with_prompt[cnt]["rag_logprob"] = prob
        # data_with_prompt[cnt]["rag_output"] = str(rag_txt)
        cnt += 1
    non_rag_prompts = [d["non_rag_prompt"] for d in data_with_prompt]
    non_rag_text = tokenizer.apply_chat_template(
        non_rag_prompts, tokenize=False, add_generation_prompt=False
    )
    non_rag_resp = llm.generate(
        non_rag_text,
        sampling_params=SamplingParams(
            temperature=0.8, top_p=0.85, max_tokens=1, prompt_logprobs=0
        ),
    )
    cnt = 0
    for non_rag_txt in non_rag_resp:
        left_idx = data_with_prompt[cnt]["non_rag_left_idx"]
        right_idx = data_with_prompt[cnt]["non_rag_right_idx"]
        prob = parse_prob(non_rag_txt, False, left_idx, right_idx)
        data_with_prompt[cnt]["non_rag_logprob"] = prob
        # data_with_prompt[cnt]["non_rag_output"] = str(non_rag_txt)
        cnt += 1
    fd = pd.DataFrame(data_with_prompt)

    fd.to_parquet(
        f"datasets/query_doc_pair/total_qwen/total_train_{cur_idx}_{cur_idx+step_len}.parquet",
        index=False,
    )


def pipeline_loop(start_len, step_len):

    file_paths = [
        "datasets/trivia_qa/total/trivia_qa_train_with_passage.parquet",
        "datasets/naturalqa/v1.0/total/nq_train_with_passage.parquet",
    ]

    raw_data = load_dataset(file_paths)

    print("raw data loaded")

    data_with_prompt = prepare_prompts(raw_data)

    fd = pd.DataFrame(data_with_prompt)
    fd["raw_idx"] = fd["raw_idx"].astype(str)
    fd.to_parquet("datasets/query_doc_pair/total_prompt.parquet", index=False)

    data_with_prompt_t = pd.read_parquet(
        "datasets/query_doc_pair/total_prompt.parquet"
    )
    data_with_prompt_t["rag_prompt"] = data_with_prompt_t["rag_prompt"].apply(list)
    data_with_prompt_t["non_rag_prompt"] = data_with_prompt_t["non_rag_prompt"].apply(
        list
    )
    data_with_prompt_t = data_with_prompt_t.to_dict("records")

    max_len = len(data_with_prompt_t)

    print("prompt prepared")

    for cur_idx in range(start_len, max_len, step_len):
        print("current loop: ", cur_idx, cur_idx + step_len)
        data_with_prompt = copy.deepcopy(
            data_with_prompt_t[cur_idx : cur_idx + step_len]
        )
        per_loop(cur_idx, step_len, data_with_prompt)


if __name__ == "__main__":
    pipeline_loop(0, 10000)
