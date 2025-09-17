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
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Constants
MAX_LEN = 512
TRAIN_BATCH_SIZE = 1
VALID_BATCH_SIZE = 1
LEARNING_RATE = 1e-05
NUM_DOCUMENTS = 25

# Initialize tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)

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
        self.classifier = torch.nn.Linear(768, 1)
        
        torch.nn.init.xavier_normal_(self.pre_classifier.weight)
        torch.nn.init.xavier_normal_(self.classifier.weight)

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
        scores = self.classifier(pooler)
        
        scores = scores.view(batch_size, num_docs)
        
        return scores

def ranknet_loss(scores):
    pos_score = scores[:, 0].unsqueeze(1)
    neg_scores = scores[:, 1:]
    
    diff = pos_score - neg_scores
    loss = torch.nn.functional.softplus(-diff)
    loss = torch.clamp(loss, min=1e-6, max=1e6)
    
    return loss.mean()

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
    save_interval = 0.01  # Save every 0.2 epochs
    total_batches = len(train_loader)
    print(total_batches)
    save_frequency = int(total_batches * save_interval)
    print(save_frequency)
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0
        correct_pairs = 0
        total_pairs = 0
        
        for batch_idx, data in enumerate(train_loader):
            # Current progress in epochs
            current_epoch = epoch + batch_idx / total_batches
            
            # Move data to device and get model outputs
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            token_type_ids = data['token_type_ids'].to(device)
            scores = model(input_ids, attention_mask, token_type_ids)
            
            # Calculate loss and update model
            loss = ranknet_loss(scores)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
            scheduler.step()
            
            # Update metrics
            running_loss += loss.item()
            pos_score = scores[:, 0].unsqueeze(1)
            neg_scores = scores[:, 1:]
            correct = (pos_score > neg_scores).float().sum().item()
            total = neg_scores.numel()
            correct_pairs += correct
            total_pairs += total

            # Print progress
            if batch_idx % 10 == 0:
                current_acc = 100 * correct_pairs / total_pairs
                print(f'Epoch: {current_epoch:.2f}, Batch: {batch_idx}, Loss: {loss.item():.4f}, '
                      f'Current Accuracy: {current_acc:.2f}%')
                print(f'Pos score: {pos_score.item():.4f}, '
                      f'Neg scores mean: {neg_scores.mean().item():.4f}')
            
            # Save checkpoint every save_interval epochs
            if (batch_idx + 1) % save_frequency == 0:
                avg_loss = running_loss / (batch_idx + 1)
                accuracy = 100 * correct_pairs / total_pairs
                
                checkpoint_path = f'checkpoints_pairwise/15pair/model_epoch_{current_epoch:.2f}_loss_{avg_loss:.4f}_acc_{accuracy:.2f}.pt'
                torch.save({
                    'epoch': current_epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'loss': avg_loss,
                    'accuracy': accuracy
                }, checkpoint_path)
                print(f'Checkpoint saved: {checkpoint_path}')

if __name__ == "__main__":
    main()