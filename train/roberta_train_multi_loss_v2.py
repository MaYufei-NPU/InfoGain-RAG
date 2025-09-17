import pandas as pd
import numpy as np
import torch
from transformers import RobertaModel, RobertaTokenizer
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import logging
import random
logging.basicConfig(level=logging.ERROR)

# Setting up the device for GPU usage
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

# Constants
MAX_LEN = 512
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
LEARNING_RATE = 1e-05
NUM_DOCUMENTS = 15
LAMBDA = 0.9  # 平衡ranknet loss和二分类loss的权重

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)

def evaluate_model(model, val_loader, device):
    model.eval()
    total_loss = 0
    total_ranknet_loss = 0
    total_bce_loss = 0
    correct_pairs = 0
    total_pairs = 0
    binary_correct = 0
    binary_total = 0
    
    with torch.no_grad():
        for data in val_loader:
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            
            rank_scores, binary_scores = model(input_ids, attention_mask, token_type_ids)
            loss, ranknet_loss, ce_loss = combined_loss(rank_scores, binary_scores)
            
            # Calculate ranking accuracy
            pos_score = rank_scores[:, 0].unsqueeze(1)
            neg_scores = rank_scores[:, 1:]
            correct = (pos_score > neg_scores).float().sum().item()
            total = neg_scores.numel()
            correct_pairs += correct
            total_pairs += total
            
            # Calculate binary classification accuracy
            binary_preds = torch.argmax(binary_scores.view(-1, 2), dim=1)
            binary_labels = torch.zeros(binary_scores.size(0), 2).long().to(binary_scores.device)
            binary_labels[:, 0] = 1
            binary_labels = binary_labels.view(-1)
            binary_correct += (binary_preds == binary_labels).float().sum().item()
            binary_total += binary_labels.size(0)
            
            total_loss += loss.item()
            total_ranknet_loss += ranknet_loss.item()
            total_bce_loss += ce_loss.item()
    
    # Calculate averages
    avg_loss = total_loss / len(val_loader)
    avg_ranknet_loss = total_ranknet_loss / len(val_loader)
    avg_bce_loss = total_bce_loss / len(val_loader)
    ranking_accuracy = 100 * correct_pairs / total_pairs
    binary_accuracy = 100 * binary_correct / binary_total
    
    return {
        'loss': avg_loss,
        'ranknet_loss': avg_ranknet_loss,
        'bce_loss': avg_bce_loss,
        'ranking_accuracy': ranking_accuracy,
        'binary_accuracy': binary_accuracy
    }

class QueryDocumentDataset(Dataset):
    def __init__(self, jsonl_file, tokenizer, max_len):
        self.data = []
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        query = item['query']
        documents = item['documents']
        
        # Get the positive document (originally first document)
        positive_doc = documents[0]
        negative_docs = documents[1:]
        
        # Randomly shuffle negative documents
        random.shuffle(negative_docs)
        
        # Put positive document at the beginning
        shuffled_documents = [positive_doc] + negative_docs
        
        # Encoding for query and all documents
        query_doc_pairs = []
        for doc in shuffled_documents:
            concatenated_text = f'query:{query} passage:{doc}'
            encoding = self.tokenizer.encode_plus(
                concatenated_text,
                add_special_tokens=True,
                max_length=self.max_len,
                padding='max_length',
                truncation=True,
                return_attention_mask=True,
                return_token_type_ids=True,
                return_tensors='pt'
            )
            query_doc_pairs.append({
                'input_ids': encoding['input_ids'].flatten(),
                'attention_mask': encoding['attention_mask'].flatten(),
                'token_type_ids': encoding['token_type_ids'].flatten()
            })

        # Stack all encodings
        input_ids = torch.stack([pair['input_ids'] for pair in query_doc_pairs])
        attention_masks = torch.stack([pair['attention_mask'] for pair in query_doc_pairs])
        token_type_ids = torch.stack([pair['token_type_ids'] for pair in query_doc_pairs])
        
        return {
            'input_ids': input_ids,
            'attention_mask': attention_masks,
            'token_type_ids': token_type_ids
        }

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
        
        input_ids = input_ids.view(-1, MAX_LEN)
        attention_mask = attention_mask.view(-1, MAX_LEN)
        token_type_ids = token_type_ids.view(-1, MAX_LEN)
        
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        
        # 获取排序得分
        rank_scores = self.rank_classifier(pooler)
        rank_scores = rank_scores.view(batch_size, num_docs)
        
        # 获取二分类得分（只对前两个样本）
        binary_features = pooler.view(batch_size, num_docs, -1)[:, :2, :]  # 只取前两个样本
        binary_scores = self.binary_classifier(binary_features)  # shape: [batch_size, 2, 2]
        
        return rank_scores, binary_scores

def combined_loss(rank_scores, binary_scores):
    # RankNet loss计算
    pos_score = rank_scores[:, 0].unsqueeze(1)
    neg_scores = rank_scores[:, 1:]
    diff = pos_score - neg_scores
    ranknet_loss = torch.nn.functional.softplus(-diff)
    ranknet_loss = torch.clamp(ranknet_loss, min=1e-6, max=1e6)
    ranknet_loss = ranknet_loss.mean()
    
    # 二分类交叉熵loss计算
    binary_labels = torch.zeros(binary_scores.size(0), 2).long().to(binary_scores.device)
    binary_labels[:, 0] = 1  # 第一个样本是正样本
    
    # binary_scores shape: [batch_size, 2, 2]
    binary_scores = binary_scores.view(-1, 2)  # reshape为[batch_size, 2]
    binary_labels = binary_labels.view(-1)  # reshape为[batch_size]
    
    ce_loss = torch.nn.CrossEntropyLoss()(binary_scores, binary_labels)
    
    # 组合loss
    total_loss = LAMBDA * ranknet_loss + (1 - LAMBDA) * ce_loss
    
    return total_loss, ranknet_loss, ce_loss

def main():
    # Set random seeds
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    
    # Load data
    dataset = QueryDocumentDataset(
        jsonl_file="/etc/ssd1/wangzihan11/rag/bert-train/data/query_documents_15.jsonl",
        tokenizer=tokenizer,
        max_len=MAX_LEN
    )
    
    # Split dataset
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=TRAIN_BATCH_SIZE,
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=VALID_BATCH_SIZE,
        shuffle=False
    )

    # Initialize model
    model = RobertaClassifier()
    model.to(device)

    # Setup training parameters
    num_epochs = 1
    num_training_steps = len(train_loader) * num_epochs
    num_warmup_steps = num_training_steps // 10

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, num_training_steps)
    max_grad_norm = 1.0

    # Training loop with more frequent checkpoints
    save_interval = 0.04
    total_batches = len(train_loader)
    save_frequency = int(total_batches * save_interval)
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        running_ranknet_loss = 0
        running_bce_loss = 0
        correct_pairs = 0
        total_pairs = 0
        
        for batch_idx, data in enumerate(train_loader):
            # Current progress in epochs
            current_epoch = epoch + batch_idx / total_batches
            
            # Move data to device and get model outputs
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            rank_scores, binary_scores = model(input_ids, attention_mask, token_type_ids)
            
            # Calculate combined loss and update model
            loss, ranknet_loss, ce_loss = combined_loss(rank_scores, binary_scores)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            running_ranknet_loss += ranknet_loss.item()
            running_bce_loss += ce_loss.item()
            pos_score = rank_scores[:, 0].unsqueeze(1)
            neg_scores = rank_scores[:, 1:]
            correct = (pos_score > neg_scores).float().sum().item()
            total = neg_scores.numel()
            correct_pairs += correct
            total_pairs += total

            binary_preds = torch.argmax(binary_scores.view(-1, 2), dim=1)
            binary_labels = torch.zeros(binary_scores.size(0), 2).long().to(binary_scores.device)
            binary_labels[:, 0] = 1
            binary_labels = binary_labels.view(-1)
            binary_correct = (binary_preds == binary_labels).float().sum().item()
            binary_total = binary_labels.size(0)

            # Print progress
            if batch_idx % 10 == 0:
                current_rank_acc = 100 * correct_pairs / total_pairs
                current_binary_acc = 100 * binary_correct / binary_total
                print(f'Epoch: {current_epoch:.2f}, Batch: {batch_idx}')
                print(f'Total Loss: {loss.item():.4f}, RankNet Loss: {ranknet_loss.item():.4f}, CE Loss: {ce_loss.item():.4f}')
                print(f'Ranking Accuracy: {current_rank_acc:.2f}%')
                print(f'Binary Classification Accuracy: {current_binary_acc:.2f}%')
                print(f'Pos score: {pos_score.item():.4f}, Neg scores mean: {neg_scores.mean().item():.4f}')
            
            # Save checkpoint
            if (batch_idx + 1) % save_frequency == 0:
                # Evaluate on validation set
                val_metrics = evaluate_model(model, val_loader, device)
                
                checkpoint_path = f'multitask_checkpoints/weight_0.9/model_epoch_{current_epoch:.2f}_valloss_{val_metrics["loss"]:.4f}_valacc_{val_metrics["ranking_accuracy"]:.2f}.pt'
                
                # Save checkpoint with validation metrics
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'val_total_loss': val_metrics['loss'],
                    'val_ranknet_loss': val_metrics['ranknet_loss'],
                    'val_bce_loss': val_metrics['bce_loss'],
                    'val_ranking_accuracy': val_metrics['ranking_accuracy'],
                    'val_binary_accuracy': val_metrics['binary_accuracy']
                }, checkpoint_path)
                
                print(f'\nCheckpoint saved: {checkpoint_path}')
                print(f'Validation Metrics:')
                print(f'Total Loss: {val_metrics["loss"]:.4f}')
                print(f'RankNet Loss: {val_metrics["ranknet_loss"]:.4f}')
                print(f'BCE Loss: {val_metrics["bce_loss"]:.4f}')
                print(f'Ranking Accuracy: {val_metrics["ranking_accuracy"]:.2f}%')
                print(f'Binary Classification Accuracy: {val_metrics["binary_accuracy"]:.2f}%\n')
                
                # Switch back to training mode
                model.train()


if __name__ == "__main__":
    main()