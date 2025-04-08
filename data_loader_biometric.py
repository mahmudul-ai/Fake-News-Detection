import pandas as pd
import numpy as np
import torch
import torch.utils.data as data
from transformers import BertTokenizer, BertModel
import json
import time

np.random.seed(42)
torch.manual_seed(42)

# Check if CUDA (GPU) is available
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load configuration from JSON file
with open('config-biometric.json', 'r') as f:
    config = json.load(f)


workspace               = config["workspace"]
metadata_each_dim       = config["metadata_each_dim"]
# trigram_embedding_dim   = config["trigram_embedding_dim"]

label_convert = {'pants-fire': 0, 'false': 1, 'barely-true': 2,
                 'half-true': 3, 'mostly-true': 4, 'true': 5}

sentiment_convert = {'NEGATIVE': 0, 'POSITIVE': 1}

# Initialize tokenizer and BERT model
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
# bert_model.eval()


def load_data(path):
    """Load and preprocess a CSV split."""
    print(f"[load_data] Loading {path}...")
    t0 = time.time()
    df = pd.read_csv(path)
    count_cols = [
        "true_counts","mostly_true_counts","half_true_counts",
        "mostly_false_counts","false_counts","pants_on_fire_counts"
    ]
    df[count_cols] = df[count_cols].fillna(0)
    df.fillna("NaN", inplace=True)
    # df['sentiment_label'] = df['sentiment_label'].map(sentiment_convert)
    print(f"[load_data] Done in {time.time()-t0:.2f}s — {len(df)} rows")
    return df


def textProcess(input_text, max_length = -1):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    if max_length == -1:
        tokens = tokenizer(input_text, truncation=True, padding=True)
    else:
        tokens = tokenizer(input_text, truncation=True, padding='max_length', max_length=max_length)
    return tokens



# def get_weighted_trigram_embeddings(frequent_trigrams, max_trigrams=5):
#     """Compute weighted CLS embeddings for top-k trigrams."""
#     print("[trigram] computing embeddings…")
#     t0 = time.time()
#     embs = []
#     for i, sample in enumerate(frequent_trigrams):
#         if i and i%1000==0:
#             print(f"  processed {i}/{len(frequent_trigrams)}")
#         topk = sample[:max_trigrams]
#         if not topk:
#             embs.append(torch.zeros(trigram_embedding_dim, device=device))
#             continue
#         texts  = [' '.join(tr[0]) for tr in topk]
#         weights= torch.tensor([tr[1] for tr in topk],
#                               dtype=torch.float, device=device)
#         weights /= weights.sum().clamp_min(1e-8)
#         enc = tokenizer(texts, padding=True, truncation=True,
#                         return_tensors='pt').to(device)
#         with torch.no_grad():
#             out = bert_model(**enc).last_hidden_state[:,0,:]  # CLS
#         embs.append((out * weights.unsqueeze(1)).sum(dim=0))
#     print(f"[trigram] done in {time.time()-t0:.2f}s")
#     return torch.stack(embs)


class LiarDataset(data.Dataset):
    def __init__(self, df, statement, label_onehot, label, justification,
                 text_meta,
                 num_meta):
        self.df = df
        self.statement  = statement
        self.label_onehot = label_onehot
        self.label      = label
        self.justification = justification

        # text metadata concatenation
        self.meta_text = torch.cat(text_meta, dim=-1)

        # numeric metadata
        # nums = [torch.tensor(col, dtype=torch.float).unsqueeze(1)
        #         for col in num_meta]
        self.meta_num = torch.cat(num_meta, dim=-1)

        # self.trigram_embeddings = trigram_embeddings

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        return (
            self.statement[idx],
            self.label_onehot[idx],
            self.label[idx],
            self.meta_text[idx],
            self.meta_num[idx],
            self.justification[idx]
        )


def build_dataset(df):
    """Tokenize, embed, and wrap a DataFrame into LiarDataset."""
    print(f"[build_dataset] building for {len(df)} rows…")
    t0 = time.time()

    # 1) text & justification
    statement = torch.tensor(textProcess(df['statement'].tolist())['input_ids'])
    justification = torch.tensor(textProcess(df['justification'].tolist())['input_ids'])

    # 2) labels
    label_onehot  = torch.nn.functional.one_hot(torch.tensor(df['label'].replace(label_convert)), num_classes=6).type(torch.float64)

    # 3) text‐metadata fields
    # text_metadata = lambda col: torch.tensor(
    #     textProcess(df[col].tolist(), metadata_each_dim)['input_ids'])
    # date, subject, speaker, speaker_description, state_info, context = map(text_metadata, [
    #     'date','subject','speaker','speaker_description','state_info','context'
    # ])
    text_cols = ['date', 'subject', 'speaker', 'speaker_description', 'state_info', 'context']
    text_metadata = [torch.tensor(textProcess(df[col].tolist(), metadata_each_dim)['input_ids']).int() for col in text_cols]

    # 4) numeric metadata columns
    num_cols = ['true_counts', 'mostly_true_counts', 'half_true_counts',
                'mostly_false_counts', 'false_counts', 'pants_on_fire_counts',
                # 'ttr', 'exclamation_count', 'adjective_count',
                # 'sentiment_label', 'sentiment_score', 'subjectivity_score',
                ]
    num_metadata = [torch.tensor(df[col].tolist(), dtype=torch.float).unsqueeze(1) for col in num_cols]

    # # 5) trigram embeddings
    # trig_emb = get_weighted_trigram_embeddings(
    #     df['frequent_trigrams'].apply(eval).tolist()
    # )

    ds = LiarDataset(
        df, statement, label_onehot, torch.tensor(df['label'].replace(label_convert)), justification,
        text_metadata,
        num_metadata
        # ,
        # trigram_embeddings=trig_emb
    )
    print(f"[build_dataset] done in {time.time()-t0:.2f}s")
    return ds


def get_dataset(split: str):
    # load and build a split by name: 'train', 'valid', or 'test'
    path = workspace + f"{split}.csv"
    df   = load_data(path)
    return build_dataset(df)


def train_loader(batch_size):
    ds = get_dataset('train')
    return data.DataLoader(ds, batch_size=batch_size, shuffle=True)

def val_loader(batch_size):
    ds = get_dataset('valid')
    return data.DataLoader(ds, batch_size=batch_size)

def test_loader(batch_size):
    ds = get_dataset('test')
    return data.DataLoader(ds, batch_size=batch_size)
