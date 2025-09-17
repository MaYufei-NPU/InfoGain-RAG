import argparse
import json
from tools import aliyun_inference,openai_inference,llama_inference,custom_inference,deepseek_inference,deepseek_r1_inference
from tqdm import tqdm
import time
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

def load_dataset(file_path):
    """Load dataset from a JSON file."""
    data = []
    try:
        with open(file_path, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        return data
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return None

def process_single_item(args):
    """Process a single data item with retries"""
    data_item, model_name, mode, idx = args  # 修改这里，添加 idx 参数
    max_retries = 5
    for _ in range(max_retries):
        try:
            if mode == 'rag':
                ans, prompt = inference_rag(data_item, model_name, 4)
            else:
                ans, prompt = inference_no_rag(data_item, model_name, 4)
            return idx, {  # 返回 idx 和处理结果
                **data_item,
                'answer': ans,
                'prompt': prompt
            }
        except Exception as e:
            print(f'Error processing item {idx}, retrying...')
            # print(f'Error details: {str(e)}')
            time.sleep(30)
    return idx, {  # 返回 idx 和空结果
                **data_item,
                'answer': 'None',
                'prompt': 'None'
            }
def inference_rag(data, model_name, top_k=4, rj='0'):
    instruction = """
    instruction:
    Based on the given passages, answer the given question.
    The answer should be as short as possible.
    Please ensure your answers are based solely on the information provided in the passages and do not rely on your personal knowledge.
    """
    shot = """
    passages:
    PASSAGE1:title: 1950 in the United States: October–December
    text:October 2 &ndash; The comic strip Peanuts by Charles M. Schulz is first published in seven U.S. newspapers.,
    PASSAGE2:title:the president of the United States
    text: In 1950, the president of the USA is Harry Truman.,
    instruction:
    Based on the given passages, answer the given question.
    The answer should be as short as possible.
    Please ensure your answers are based solely on the information provided in the passages and do not rely on your personal knowledge.
    question:Who was President when the first Peanuts cartoon was published?
    answer:
    """

    qs = data["question"]
    merged_passage = ""
    passages = data['top_passages']
    for j in range(top_k):
        tmp_passage = 'title:' + passages[j]['title'] + 'text:' + passages[j]['text']
        merged_passage += f"PASSAGE{j+1}:" + tmp_passage + "\n"
    prompt_template = "passages:\n" + merged_passage + instruction + "\nquestion:\n"
    prompt = prompt_template + qs

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": shot},
        {"role": "assistant", "content": "Harry Truman"},
        {"role": "user", "content": prompt},
    ]
    if model_name in ('meta-llama/Llama-2-13b-chat-hf','google/gemma-2-27b-it','google/gemma-2b-it','google/gemma-2-9b-it','anthropic/claude-3.5-sonnet-20241022','google/gemini-2.5-pro-preview-05-06','gpt-4.1-2025-04-14'):
        if model_name == 'anthropic/claude-3.5-sonnet-20241022':
            messages = [
                {"role": "user", "content": shot},
                {"role": "assistant", "content": "Harry Truman"},
                {"role": "user", "content": prompt},
            ]
            ans = custom_inference(model_name, messages) 
        else:
            ans = custom_inference(model_name, messages)
    elif 'r1' in model_name.lower():
        ans = deepseek_r1_inference(model_name, messages)
    elif 'gpt' in model_name.lower():
        ans = openai_inference(model_name, messages)
    elif 'deepseek' in model_name.lower():
        ans = deepseek_inference(model_name, messages)
    elif 'llama' in model_name.lower():
        ans = llama_inference(model_name, messages)
    else:
        ans = aliyun_inference(model_name, messages)
    return ans, prompt

def inference_no_rag(data, model_name, top_k=4, rj='0'):
    instruction = """
    instruction:
    answer the given question.
    The answer should be as short as possible.
    """
    shot = """
    instruction:
    answer the given question.
    The answer should be as short as possible.
    question:Who was President when the first Peanuts cartoon was published?
    answer:
    """

    qs = data["question"]
    prompt_template = instruction + "\nquestion:\n"
    prompt = prompt_template + qs

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": shot},
        {"role": "assistant", "content": "Harry Truman"},
        {"role": "user", "content": prompt},
    ]

    if model_name in ('meta-llama/Llama-2-13b-chat-hf','google/gemma-2-27b-it','google/gemma-2b-it','google/gemma-2-9b-it','anthropic/claude-3.5-sonnet-20241022'):
        if model_name == 'anthropic/claude-3.5-sonnet-20241022':
            messages = [
                {"role": "user", "content": shot},
                {"role": "assistant", "content": "Harry Truman"},
                {"role": "user", "content": prompt},
            ]
            ans = custom_inference(model_name, messages) 
        else:
            ans = custom_inference(model_name, messages)
    elif 'r1' in model_name.lower():
        ans = deepseek_r1_inference(model_name, messages)
    elif 'gpt' in model_name.lower():
        ans = openai_inference(model_name, messages)
    elif 'deepseek' in model_name.lower():
        ans = deepseek_inference(model_name, messages)
    elif 'llama' in model_name.lower():
        ans = llama_inference(model_name, messages)
    else:
        ans = aliyun_inference(model_name, messages)
    return ans, prompt

def save_to_jsonl(data, file_path):
    """Save data to a JSONL file."""
    try:
        with open(file_path, 'w') as file:
            for entry in data:
                if entry is not None:  # Only write non-None entries
                    file.write(json.dumps(entry) + '\n')
    except Exception as e:
        print(f"Error saving to JSONL file: {e}")

def main():
    parser = argparse.ArgumentParser(description="Process and judge a dataset with a specified model and rule.")
    parser.add_argument('--qa_dataset', type=str, required=True, help="Path to the dataset file json.")
    parser.add_argument('--inference_model', type=str, required=True, help="Name of the model to use for processing.")
    parser.add_argument('--output_file', type=str, required=True, help="Path to the output JSONL file.")
    parser.add_argument('--num_workers', type=int, default=4, help="Number of parallel workers")
    parser.add_argument('--mode', type=str, choices=['rag', 'no_rag'], required=True, help="Mode of inference")

    args = parser.parse_args()

    # Load dataset
    data = load_dataset(args.qa_dataset)
    if data is None:
        return

    # Prepare data for parallel processing
    data = data[:500]  # Limit to 500 items
    process_args = [(item, args.inference_model, args.mode, idx) for idx, item in enumerate(data)]

    # Process data in parallel
    results_dict = {}
    executor = ProcessPoolExecutor(max_workers=args.num_workers)
    
    try:
        futures = [executor.submit(process_single_item, arg) for arg in process_args]
        
        for future in tqdm(as_completed(futures), total=len(data)):
            idx, result = future.result()
            results_dict[idx] = result
            
    except Exception as e:
        print(f"Error during execution: {e}")
    finally:
        # 显式关闭执行器
        executor.shutdown(wait=True)
        # 确保所有进程都被终止
        for _ in range(3):  # 尝试多次确保清理
            if hasattr(executor, '_processes') and executor._processes:
                time.sleep(1)
                executor._processes.clear()
        
        # 清理所有剩余进程
        # cleanup_processes()


    # 按原始顺序重组结果
    processed_data = []
    for i in range(len(data)):
        if results_dict.get(i) is not None:
            processed_data.append(results_dict[i])

    # Save results
    save_to_jsonl(processed_data, args.output_file)

if __name__ == "__main__":
    multiprocessing.freeze_support()  # Required for Windows
    main()