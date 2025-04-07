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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased').to(device)
bert_model.eval()


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
    df['sentiment_label'] = df['sentiment_label'].map(sentiment_convert)
    print(f"[load_data] Done in {time.time()-t0:.2f}s — {len(df)} rows")
    return df


def textProcess(input_text, max_length = -1):
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
                 date, subject, speaker, speaker_desc, state_info, context,
                 *num_meta):
        self.statement  = statement
        self.label_onehot = label_onehot
        self.label      = label
        self.justification = justification

        # text metadata concatenation
        self.meta_text = torch.cat([
            date.int(), subject.int(), speaker.int(),
            speaker_desc.int(), state_info.int(), context.int()
        ], dim=-1)

        # numeric metadata
        nums = [torch.tensor(col, dtype=torch.float, device=device).unsqueeze(1)
                for col in num_meta]
        self.meta_num = torch.cat(nums, dim=-1)

        # self.trigram_embeddings = trigram_embeddings

    def __len__(self):
        return self.statement.size(0)

    def __getitem__(self, i):
        return (
            self.statement[i],
            self.label_onehot[i],
            self.label[i],
            self.meta_text[i],
            self.meta_num[i],
            self.justification[i]
            # ,
            # self.trigram_embeddings[i]
        )


def build_dataset(df):
    """Tokenize, embed, and wrap a DataFrame into LiarDataset."""
    print(f"[build_dataset] building for {len(df)} rows…")
    t0 = time.time()

    # 1) text & justification
    stmt = torch.tensor(textProcess(df['statement'].tolist())['input_ids'],
                        device=device)
    just = torch.tensor(textProcess(df['justification'].tolist())['input_ids'],
                        device=device)

    # 2) labels
    lbl  = torch.tensor(df['label'].replace(label_convert).tolist(),
                        device=device)
    lbl1 = torch.nn.functional.one_hot(lbl, num_classes=6).float()

    # 3) text‐metadata fields
    tm = lambda col: torch.tensor(
        textProcess(df[col].tolist(), metadata_each_dim)['input_ids'],
        device=device
    )
    date, subj, spk, spk_desc, st_info, ctx = map(tm, [
        'date','subject','speaker','speaker_description','state_info','context'
    ])

    # 4) numeric metadata columns
    num_cols = [
        df['true_counts'].tolist(),
        df['mostly_true_counts'].tolist(),
        df['half_true_counts'].tolist(),
        df['mostly_false_counts'].tolist(),
        df['false_counts'].tolist(),
        df['pants_on_fire_counts'].tolist(),
        df['ttr'].tolist(),
        df['exclamation_count'].tolist(),
        df['adjective_count'].tolist(),
        # df['sentiment_label'].tolist(),
        # df['sentiment_score'].tolist(),
        # df['subjectivity_score'].tolist(),
        # df['contradiction_score'].tolist(),
    ]

    # # 5) trigram embeddings
    # trig_emb = get_weighted_trigram_embeddings(
    #     df['frequent_trigrams'].apply(eval).tolist()
    # )

    ds = LiarDataset(
        df, stmt, lbl1, lbl, just,
        date, subj, spk, spk_desc, st_info, ctx,
        *num_cols
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
