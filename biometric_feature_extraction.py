import json
import pandas as pd
import re
import nltk
from nltk import ngrams
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import spacy
from collections import Counter
import torch
from transformers import pipeline
from itertools import combinations

torch.manual_seed(42)


# Check if CUDA is available
device = 0 if torch.cuda.is_available() else -1
print(f"The code is running using '{device}'")

nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt_tab')
nltk.download('averaged_perceptron_tagger_eng')

nlp = spacy.load('en_core_web_sm')  # Load spaCy's English model


stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Load configuration from JSON file
with open('config-biometric.json', 'r') as f:
    config = json.load(f)

workspace = config["workspace"]

col = ["id", "label", "statement", "date", "subject", "speaker", "speaker_description", "state_info", "true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts", "context", "justification"]


train_data = pd.read_csv(workspace + 'train.csv')
test_data = pd.read_csv(workspace + 'test.csv')
val_data = pd.read_csv(workspace + 'valid.csv')

# # Replace NaN values with 'NaN'
# train_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = train_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
# train_data.fillna('NaN', inplace=True)

# test_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = test_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
# test_data.fillna('NaN', inplace=True)

# val_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]] = val_data[["true_counts", "mostly_true_counts", "half_true_counts", "mostly_false_counts", "false_counts", "pants_on_fire_counts"]].fillna(0)
# val_data.fillna('NaN', inplace=True)


def clean_text(text):
    # Remove HTML tags
    text = re.sub('<.*?>', '', text)

    # Remove special characters and numbers
    text = re.sub('[^a-zA-Z\s]', '', text)

    # Convert to lowercase
    text = text.lower()

    # Remove stopwords and lemmatize
    words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]

    # Remove redundant whitespaces
    text = ' '.join(words)

    return text


# train_data['speaker_description'] = train_data['speaker_description'].apply(clean_text)
# train_data['justification'] = train_data['justification'].apply(clean_text)

# # X_test['statement'] = X_test['statement'].apply(clean_text)
# test_data['speaker_description'] = test_data['speaker_description'].apply(clean_text)
# test_data['justification'] = test_data['justification'].apply(clean_text)

# # X_val['statement'] = X_val['statement'].apply(clean_text)
# val_data['speaker_description'] = val_data['speaker_description'].apply(clean_text)
# val_data['justification'] = val_data['justification'].apply(clean_text)


def analyze_text(text, clean = True, inplace=False):
    # Count exclamation marks
    exclamation_count = text.count('!')

    if clean:
      text = clean_text(text)

    # Tokenize the text
    tokens = nltk.word_tokenize(text)
    # Perform N-gram analysis (e.g., trigrams)
    n = 3  # Adjust 'n' for different N-grams
    trigrams = list(ngrams(tokens, n))
    trigram_counts = Counter(trigrams)

    # Identify frequent N-grams (example: top 5)
    frequent_trigrams = trigram_counts.most_common(5)

    # Calculate lexical diversity (Type-Token Ratio)
    types = set(tokens)
    ttr = len(types) / len(tokens) if tokens else 0

    # Perform part-of-speech tagging
    pos_tags = nltk.pos_tag(tokens)

    # Find adjective-heavy structures
    adjective_count = sum(1 for word, tag in pos_tags if tag.startswith('JJ'))

    out = {
          "frequent_trigrams": frequent_trigrams,
          "ttr": ttr,
          "exclamation_count": exclamation_count,
          "adjective_count": adjective_count,
      }
    if inplace:
      out['statement-clean'] = text
      # print(text)
    return out

# Apply the analysis to the 'statement' column and assign the results to new columns
out_cols = ['statement-clean', 'frequent_trigrams', 'ttr', 'exclamation_count', 'adjective_count']
out = train_data['statement'].apply(analyze_text, inplace=True).apply(pd.Series)
for col in out_cols:
  train_data[col] = out[col]

out = train_data['statement'].apply(analyze_text, inplace=True).apply(pd.Series)
for col in out_cols:
  test_data[col] = out[col]

out = val_data['statement'].apply(analyze_text, inplace=True).apply(pd.Series)
for col in out_cols:
  val_data[col] = out[col]

# Save DataFrames and labels as CSV files
train_data.to_csv(workspace +'train.csv', index=False)
test_data.to_csv(workspace + 'test.csv', index=False)
val_data.to_csv(workspace + 'valid.csv', index=False)

print("Checkpoint: Linguistic features added!")


# # Sentiment Analysis

# Load the sentiment analysis pipeline with GPU support
sentiment_classifier = pipeline("sentiment-analysis", device=device)

# Define a subjectivity detection function (this should be replaced with an actual transformer-based model)
def subjectivity_detector(text):
    # Placeholder for subjectivity detection using a transformer-based model
    # This can be replaced with a fine-tuned transformer model running on GPU
    if "opinion" in text.lower() or "believe" in text.lower():
        return 0.8
    else:
        return 0.3

# Function for sentiment and subjectivity analysis
def sentiment_analysis(text, clean=True):
    if clean:
        text = clean_text(text)

    # Sentiment analysis (on GPU)
    sentiment_result = sentiment_classifier(text)[0]
    sentiment_label = sentiment_result['label']
    sentiment_score = sentiment_result['score']

    # Subjectivity detection (currently a placeholder)
    subjectivity_score = subjectivity_detector(text)

    return {
        "sentiment_label": sentiment_label,
        "sentiment_score": sentiment_score,
        "subjectivity_score": subjectivity_score,
    }

# Apply analysis efficiently
def analyze_dataframe(df, text_column):
    results = df[text_column].apply(sentiment_analysis).apply(pd.Series)
    df[['sentiment_label', 'sentiment_score', 'subjectivity_score']] = results
    return df

# Apply to datasets
train_data = analyze_dataframe(train_data, 'statement-clean')
test_data = analyze_dataframe(test_data, 'statement-clean')
val_data = analyze_dataframe(val_data, 'statement-clean')


# Save DataFrames and labels as CSV files
train_data.to_csv(workspace +'train.csv', index=False)
test_data.to_csv(workspace + 'test.csv', index=False)
val_data.to_csv(workspace + 'valid.csv', index=False)

print("Checkpoint: Sentiment features added")

# # Load preprocessed datasets
# train_data = pd.read_csv(workspace +'train.csv')
# test_data = pd.read_csv(workspace +'test.csv')
# val_data = pd.read_csv(workspace +'valid.csv')

## Contradiction Score

# Load the RoBERTa model for Natural Language Inference (NLI)

nli_classifier = pipeline("text-classification", model="roberta-large-mnli", device=device)

def extract_clauses(sentence):
    """
    Extracts clauses from a sentence using POS tagging and dependency parsing.

    Args:
    - sentence (str): The input sentence.

    Returns:
    - list: Extracted clauses.
    """
    doc = nlp(sentence.lower())
    # text = ' '.join(words)
    clauses = []
    clause = []

    for token in doc:
        clause.append(token.text)

        # Clause boundary indicators: punctuations (commas, semicolons) and conjunctions
        if token.dep_ in {"cc", "mark", "punct", "relcl"} or token.text in {",", ";", ".", "and", "but", "because", "although", "if", "when", "while", "hence"}:
            if clause:
                clauses.append(" ".join(clause).strip())
                clause = []

    # Add any remaining clause
    if clause:
        clauses.append(" ".join(clause).strip())

    return clauses

def detect_contradictions(statement):
  """
  Detect contradiction between two sentences using RoBERTa-large-MNLI.

  Args:
  - premise (str): The context sentence.
  - hypothesis (str): The reply sentence.

  Returns:
  - dict: Contains label ('ENTAILMENT', 'NEUTRAL', or 'CONTRADICTION') and confidence score.
  """
  clauses = extract_clauses(statement)
#   print(len(clauses))

  # Generate all possible clause pairs
  clause_pairs = list(combinations(clauses, 2))

  contradiction_count = 0
  total_pairs = len(clause_pairs)

  for premise, hypothesis in clause_pairs:
    # print(f"🧐 Premise: {premise} | Hypothesis: {hypothesis}")
    result = nli_classifier(f"{premise} </s> {hypothesis}")
    label_scores = {res["label"]: res["score"] for res in result}
    contradiction_score = label_scores.get("CONTRADICTION", 0.0)

    out = {
        "premise": premise,
        "hypothesis": hypothesis,
        "label": max(label_scores, key=label_scores.get),
        "contradiction_score": contradiction_score
    }
    # print(out["label"])

    if out["label"].lower() == 'contradiction':
        contradiction_count += 1

  contradiction_score = contradiction_count / total_pairs if total_pairs > 0 else 0

  return {"contradiction_score": contradiction_score}



train_data["contradiction_score"] = train_data["statement"].apply(detect_contradictions).apply(pd.Series)
test_data["contradiction_score"] = test_data["statement"].apply(detect_contradictions).apply(pd.Series)
val_data["contradiction_score"] = val_data["statement"].apply(detect_contradictions).apply(pd.Series)

# Save DataFrames and labels as CSV files
train_data.to_csv(workspace +'train.csv', index=False)
test_data.to_csv(workspace + 'test.csv', index=False)
val_data.to_csv(workspace + 'valid.csv', index=False)

print("Checkpoint: Contradiction scores added")