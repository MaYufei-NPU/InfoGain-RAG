import argparse
import json
from tqdm import tqdm
import torch
from transformers import RobertaTokenizer, RobertaModel, RobertaConfig
import torch.nn as nn

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)
MAX_LEN = 512
class RobertaClassifier(torch.nn.Module):
    def __init__(self):
        super(RobertaClassifier, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-large")
        self.pre_classifier = torch.nn.Linear(1024, 768)
        self.dropout = torch.nn.Dropout(0.3)
        # 排序得分的分类头
        self.rank_classifier = torch.nn.Linear(768, 1)
        # 二分类的分类头
        self.binary_classifier = torch.nn.Linear(768, 2)
        
        # 初始化权重
        torch.nn.init.xavier_normal_(self.pre_classifier.weight)
        torch.nn.init.xavier_normal_(self.rank_classifier.weight)
        torch.nn.init.xavier_normal_(self.binary_classifier.weight)

    def forward(self, input_ids, attention_mask, token_type_ids):
        batch_size = input_ids.size(0)
        num_docs = input_ids.size(1)
        
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        
        # 获取排序得分
        rank_scores = self.rank_classifier(pooler)
        # rank_scores = rank_scores.view(batch_size, num_docs)
        
        # 获取二分类得分（只对前两个样本）
        # binary_features = pooler.view(batch_size, num_docs, -1)[:, :2, :]  # 只取前两个样本
        # binary_scores = self.binary_classifier(binary_features)  # shape: [batch_size, 2, 2]
        
        return rank_scores, 0

def batch_inference(model, query_list, passage_list, batch_size=10):
    model.eval()
    scores = []
    
    for i in range(0, len(query_list), batch_size):
        batch_queries = query_list[i:i + batch_size]
        batch_passages = passage_list[i:i + batch_size]
        
        concatenated_texts = [
            f'query:{query} passage:{passage}' 
            for query, passage in zip(batch_queries, batch_passages)
        ]
        
        encoding = tokenizer(
            concatenated_texts,
            add_special_tokens=True,
            max_length=512,
            padding=True,
            truncation=True,
            return_attention_mask=True,
            return_token_type_ids=True,
            return_tensors='pt'
        )
        
        input_ids = encoding['input_ids'].to(device)
        attention_mask = encoding['attention_mask'].to(device)
        token_type_ids = encoding['token_type_ids'].to(device)
        
        with torch.no_grad():
            outputs,_ = model(input_ids, attention_mask, token_type_ids)
            batch_scores = outputs.squeeze(-1).cpu().numpy()
            scores.extend(batch_scores.tolist())
    
    return scores

def main(model_path, input_file, output_file):
    model = RobertaClassifier()
    checkpoint = torch.load(model_path)
    model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.to(device)
    model.eval()
    
    data = []
    with open(input_file, 'r') as f:
        for line in f:
            data.append(json.loads(line))
            if len(data) == 500:
                break
    # 对每个问题进行重排序
    for item in tqdm(data[300:500]):
        question = item['question']
        passages = item['top_passages']
        
        # 准备批处理数据
        queries = [question] * len(passages)
        passage_texts = [f"{p['title']} {p['text']}" for p in passages]
        
        # 批量计算分数
        scores = batch_inference(model, queries, passage_texts)
        
        # 为每个passage分配分数
        for passage, score in zip(passages, scores):
            passage['score'] = float(score)
        
        # 根据分数降序排序
        item['top_passages'] = sorted(passages, key=lambda x: x['score'], reverse=True)
    
    with open(output_file, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + '\n')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Rerank passages using fine-tuned model.')
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_file', type=str, required=True)
    parser.add_argument('--output_file', type=str, required=True)
    
    args = parser.parse_args()
    main(args.model_path, args.input_file, args.output_file)