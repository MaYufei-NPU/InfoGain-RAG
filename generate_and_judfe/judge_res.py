import json
import re
import argparse

def regularization(input_string):
    result = re.sub(r'[.,]', '', input_string)
    result = re.sub(r'[\"\']', '', input_string)
    result = result.split('\n')[0]
    result = result.lower()
    result = result.split(" was a ", 1)[1].strip() if " was a " in result else result
    result = result.split(" was an ", 1)[1].strip() if " was an " in result else result
    result = result.split(" was ", 1)[1].strip() if " was " in result else result
    result = result.split("the answer is", 1)[1].strip() if "the answer is" in result else result
    result = result.lower().strip()
    return result

def has_answer_loose(output, answer_list):
    for answer in answer_list:
        if answer in output or output in answer:
            return True
    return False

def main(test_path, res_path):
    test_list = []
    with open(test_path, 'r') as file:
        for line in file:
            test_list.append(json.loads(line))

    res_list = []
    with open(res_path, 'r') as file:
        for line in file:
            res_list.append(json.loads(line))

    cnt_answer = 0

    for i in range(500):
        try:
            if has_answer_loose(regularization(res_list[i]['answer']), [answer.lower().strip() for answer in test_list[i]['answers']]):
                cnt_answer += 1
        except Exception as e:
            print(f"Error at index {i}: {e}")

    print('EM:', cnt_answer)
    print("-" * 50)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate answers from a JSONL file.")
    parser.add_argument('test_path', type=str, help='Path to the test JSONL file.')
    parser.add_argument('res_path', type=str, help='Path to the results JSONL file.')

    args = parser.parse_args()
    main(args.test_path, args.res_path)