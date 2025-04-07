import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from transformers import BertTokenizer
import json

# Load configuration from JSON file
with open('config.json', 'r') as f:
    config = json.load(f)

np.random.seed(42)
torch.manual_seed(42)

workspace = config["workspace"]
metadata_each_dim = config["metadata_each_dim"]


col = ["id", "label", "statement", "date", "subject", "speaker", "speaker_description", "state_info", "true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts", "context", "justification"]

label_map = {0: 'pants-fire', 1: 'false', 2: 'barely-true', 3: 'half-true', 4: 'mostly-true', 5: 'true'}
label_convert = {'pants-fire': 0, 'false': 1, 'barely-true': 2, 'half-true': 3, 'mostly-true': 4, 'true':5}

train_data = pd.read_csv(workspace + 'train.csv')
test_data = pd.read_csv(workspace + 'test.csv')
val_data = pd.read_csv(workspace + 'valid.csv')

# Replace NaN values with 'NaN'
train_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = train_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
train_data.fillna('NaN', inplace=True)

test_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = test_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
test_data.fillna('NaN', inplace=True)

val_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = val_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
val_data.fillna('NaN', inplace=True)


def textProcess(input_text, max_length = -1):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if max_length == -1:
        tokens = tokenizer(input_text, truncation=True, padding=True)
    else:
        tokens = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length)
    return tokens


# Define a custom dataset for loading the data
class LiarDataset(data.Dataset):
    def __init__(self, data_df, statement, label_onehot, label, date, subject, speaker, speaker_description, state_info,
                    true_counts, mostly_true_counts, half_true_counts, mostly_false_counts,
                    false_counts, pants_on_fire_counts, context, justification):
        self.data_df = data_df
        self.statement = statement
        self.label_onehot = label_onehot
        self.label = label
        self.justification = justification
        self.metadata_text = torch.cat((date.int(), subject.int(), speaker.int(), speaker_description.int(), state_info.int(), context.int()), dim=-1)
        self.metadata_number = torch.cat((torch.tensor(true_counts, dtype=torch.float).unsqueeze(1), torch.tensor(mostly_true_counts, dtype=torch.float).unsqueeze(1), 
                                   torch.tensor(half_true_counts, dtype=torch.float).unsqueeze(1), torch.tensor(mostly_false_counts, dtype=torch.float).unsqueeze(1), 
                                   torch.tensor(false_counts, dtype=torch.float).unsqueeze(1), torch.tensor(pants_on_fire_counts, dtype=torch.float).unsqueeze(1)), dim=-1)

    def __len__(self):
        return len(self.data_df)
    
    def __getitem__(self, idx):
        statement = self.statement[idx]
        label_onehot = self.label_onehot[idx]
        label = self.label[idx]
        justification = self.justification[idx]
        metadata_text = self.metadata_text[idx]
        metadata_number = self.metadata_number[idx]
        return statement, label_onehot, label, metadata_text, metadata_number, justification


# Define the data loaders for training and validation
train_text = torch.tensor(textProcess(train_data['statement'].tolist())['input_ids'])
train_justification = torch.tensor(textProcess(train_data['justification'].tolist())['input_ids'])
train_label = torch.nn.functional.one_hot(torch.tensor(train_data['label'].replace(label_convert)), num_classes=6).type(torch.float64)
train_date = torch.tensor(textProcess(train_data['date'].tolist(), metadata_each_dim)['input_ids'])
train_subject = torch.tensor(textProcess(train_data['subject'].tolist(), metadata_each_dim)['input_ids'])
train_speaker = torch.tensor(textProcess(train_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
train_speaker_description = torch.tensor(textProcess(train_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
train_state_info = torch.tensor(textProcess(train_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
train_context = torch.tensor(textProcess(train_data['context'].tolist(), metadata_each_dim)['input_ids'])

train_dataset = LiarDataset(train_data, train_text, train_label, torch.tensor(train_data['label'].replace(label_convert)), 
                            train_date, train_subject, train_speaker, train_speaker_description, train_state_info, 
                            train_data['true_counts'].tolist(), train_data['mostly_true_counts'].tolist(), 
                            train_data['half_true_counts'].tolist(), train_data['mostly_false_counts'].tolist(), 
                            train_data['false_counts'].tolist(), train_data['pants_on_fire_counts'].tolist(), train_context, train_justification)

def train_loader(batch_size):
    return data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

val_text = torch.tensor(textProcess(val_data['statement'].tolist())['input_ids'])
val_justification = torch.tensor(textProcess(val_data['justification'].tolist())['input_ids'])
val_label = torch.nn.functional.one_hot(torch.tensor(val_data['label'].replace(label_convert)), num_classes=6).type(torch.float64)
val_date = torch.tensor(textProcess(val_data['date'].tolist(), metadata_each_dim)['input_ids'])
val_subject = torch.tensor(textProcess(val_data['subject'].tolist(), metadata_each_dim)['input_ids'])
val_speaker = torch.tensor(textProcess(val_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
val_speaker_description = torch.tensor(textProcess(val_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
val_state_info = torch.tensor(textProcess(val_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
val_context = torch.tensor(textProcess(val_data['context'].tolist(), metadata_each_dim)['input_ids'])

val_dataset = LiarDataset(val_data, val_text, val_label, torch.tensor(val_data['label'].replace(label_convert)),
                          val_date, val_subject, val_speaker, val_speaker_description, val_state_info, 
                          val_data['true_counts'].tolist(), val_data['mostly_true_counts'].tolist(), 
                          val_data['half_true_counts'].tolist(), val_data['mostly_false_counts'].tolist(), 
                          val_data['false_counts'].tolist(), val_data['pants_on_fire_counts'].tolist(), val_context, val_justification)

def val_loader(batch_size): 
    return data.DataLoader(val_dataset, batch_size=batch_size)

test_text = torch.tensor(textProcess(test_data['statement'].tolist())['input_ids'])
test_justification = torch.tensor(textProcess(test_data['justification'].tolist())['input_ids'])
test_label = torch.nn.functional.one_hot(torch.tensor(test_data['label'].replace(label_convert)), num_classes=6).type(torch.float64)
test_date = torch.tensor(textProcess(test_data['date'].tolist(), metadata_each_dim)['input_ids'])
test_subject = torch.tensor(textProcess(test_data['subject'].tolist(), metadata_each_dim)['input_ids'])
test_speaker = torch.tensor(textProcess(test_data['speaker'].tolist(), metadata_each_dim)['input_ids'])
test_speaker_description = torch.tensor(textProcess(test_data['speaker_description'].tolist(), metadata_each_dim)['input_ids'])
test_state_info = torch.tensor(textProcess(test_data['state_info'].tolist(), metadata_each_dim)['input_ids'])
test_context = torch.tensor(textProcess(test_data['context'].tolist(), metadata_each_dim)['input_ids'])

test_dataset = LiarDataset(test_data, test_text, test_label, torch.tensor(test_data['label'].replace(label_convert)),
                          test_date, test_subject, test_speaker, test_speaker_description, test_state_info, 
                          test_data['true_counts'].tolist(), test_data['mostly_true_counts'].tolist(), 
                          test_data['half_true_counts'].tolist(), test_data['mostly_false_counts'].tolist(), 
                          test_data['false_counts'].tolist(), test_data['pants_on_fire_counts'].tolist(), test_context, test_justification)

def test_loader(batch_size): 
    return data.DataLoader(test_dataset, batch_size=batch_size)