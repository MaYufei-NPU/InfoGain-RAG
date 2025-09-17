# Importing the libraries needed
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import torch
import transformers
import json
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from transformers import RobertaModel, RobertaTokenizer
import logging
logging.basicConfig(level=logging.ERROR)
import pandas as pd
import random

# Setting up the device for GPU usage

from torch import cuda
device = 'cuda' if cuda.is_available() else 'cpu'

# Defining some key variables that will be used later on in the training
MAX_LEN = 512
TRAIN_BATCH_SIZE = 8
VALID_BATCH_SIZE = 4
# EPOCHS = 1
LEARNING_RATE = 1e-05
tokenizer = RobertaTokenizer.from_pretrained('roberta-large', truncation=True, do_lower_case=True)

file_path = '/etc/ssd1/wangzihan11/FlagEmbedding/examples/finetune/reranker/example_data/total_train_prob_celoss_strategy1.parquet'
df = pd.read_parquet(file_path)

# Assuming df is your DataFrame
queries_and_passages = []
labels = []

# Pre-generate random indices for sampling
# random_indices = np.random.randint(0, len(df), size=len(df))

# for i in tqdm(range(len(df))):
#     line = df.iloc[i]
#     if int(line['label']) == 1:
#         concatenated_text = 'query:' + line['query'] + 'passage:' + line['passage_title'] + line['passage_text']
#         queries_and_passages.append(concatenated_text)
#         labels.append(int(line['label']))
#     else:
#         # Use pre-generated random index to sample a line
#         random_index = random_indices[i]
#         sampled_line = df.iloc[random_index]
#         concatenated_text = 'query:' + line['query'] + 'passage:' + sampled_line['passage_title'] + sampled_line['passage_text']
#         queries_and_passages.append(concatenated_text)
#         labels.append(int(line['label']))

for i in tqdm(range(len(df))):
    line = df.iloc[i]
    concatenated_text = 'query:' + line['query'] + ' passage:' + line['passage_title'] + line['passage_text']
    queries_and_passages.append(concatenated_text)
    labels.append(int(line['label']))

# Convert to DataFrame for easier manipulation
result_df = pd.DataFrame({
    'text': queries_and_passages,
    'label': labels
})

# Separate the DataFrame by label
label_1_df = result_df[result_df['label'] == 1]
label_0_df = result_df[result_df['label'] == 0]

# Determine the minimum number of samples between the two labels
min_samples = min(len(label_1_df), len(label_0_df))

# Randomly sample from both DataFrames to balance the dataset
balanced_label_1_df = label_1_df.sample(n=min_samples, random_state=42)
balanced_label_0_df = label_0_df.sample(n=min_samples, random_state=42)

# Concatenate the balanced DataFrames
balanced_df = pd.concat([balanced_label_1_df, balanced_label_0_df]).reset_index(drop=True)

# Now you have a balanced DataFrame
queries_and_passages = balanced_df['text'].tolist()
labels = balanced_df['label'].tolist()


# Combine the two lists into a list of tuples and shuffle them
combined_data = list(zip(queries_and_passages, labels))
random.shuffle(combined_data)

# Unzip the shuffled data back into two lists
texts, labels = zip(*combined_data)
print(len(texts), len(labels))


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_len,
            return_token_type_ids=True,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'text': text,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long),
            'token_type_ids': encoding['token_type_ids'].flatten()
        }

max_len = 512
batch_size = 16

train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_len)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_len)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class RobertaClass(torch.nn.Module):
    def __init__(self):
        super(RobertaClass, self).__init__()
        self.l1 = RobertaModel.from_pretrained("roberta-large")
        self.pre_classifier = torch.nn.Linear(1024, 768)
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, 2)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = torch.nn.ReLU()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output

model = RobertaClass()
model.to(device)

# Creating the loss function and optimizer
loss_function = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params =  model.parameters(), lr=LEARNING_RATE)


def calcuate_accuracy(preds, targets):
    n_correct = (preds==targets).sum().item()
    return n_correct

# Defining the training function on the 80% of the dataset for tuning the distilbert model

def train(epoch):
    tr_loss = 0
    n_correct = 0
    nb_tr_steps = 0
    nb_tr_examples = 0
    model.train()
    for _,data in tqdm(enumerate(train_loader, 0)):
        ids = data['input_ids'].to(device, dtype = torch.long)
        mask = data['attention_mask'].to(device, dtype = torch.long)
        token_type_ids = data['token_type_ids'].to(device, dtype = torch.long)
        targets = data['label'].to(device, dtype = torch.long)

        outputs = model(ids, mask, token_type_ids)
        loss = loss_function(outputs, targets)
        print(loss)
        tr_loss += loss.item()
        big_val, big_idx = torch.max(outputs.data, dim=1)
        n_correct += calcuate_accuracy(big_idx, targets)

        nb_tr_steps += 1
        nb_tr_examples+=targets.size(0)
        
        if _!=0 and _%500==0:
            loss_step = tr_loss/nb_tr_steps
            accu_step = (n_correct*100)/nb_tr_examples 
            print(f"Training Loss per 5000 steps: {loss_step}")
            print(f"Training Accuracy per 5000 steps: {accu_step}")
            
            save_path = f'new_checkpoints/model_epoch{epoch}_step{_}.pt'
            torch.save(model.state_dict(), save_path)
            print(f"Model weights saved to {save_path}")
        
        optimizer.zero_grad()
        loss.backward()
        # # When using GPU
        optimizer.step()

    print(f'The Total Accuracy for Epoch {epoch}: {(n_correct*100)/nb_tr_examples}')
    epoch_loss = tr_loss/nb_tr_steps
    epoch_accu = (n_correct*100)/nb_tr_examples
    print(f"Training Loss Epoch: {epoch_loss}")
    print(f"Training Accuracy Epoch: {epoch_accu}")

    return 

EPOCHS = 1
for epoch in range(EPOCHS):
    train(epoch)