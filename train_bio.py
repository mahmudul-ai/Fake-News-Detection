import time
import torch
import torch.nn as nn
from model_bio import LiarModel
import logging
from sklearn.metrics import f1_score
from data_loader_biometric import train_loader, val_loader
import json

# Load configuration from JSON file
with open('config-biometric.json', 'r') as f:
    config = json.load(f)

# Access configuration values
workspace = config["workspace"]
model_save = config["model_save"]
# model_name = config["model_name"]
num_epochs = config["num_epochs"]
batch_size = config["batch_size"]
learning_rate = config["learning_rate"]
# num_classes = config["num_classes"]
padding_idx = config["padding_idx"]


vocab_size = config["vocab_size"]
embedding_dim = config["embedding_dim"]
n_filters = config["n_filters"]
filter_sizes = config["filter_sizes"]
output_dim = config["output_dim"]
dropout = config["dropout"]
input_dim = config["input_dim"]
input_dim_metadata = config["input_dim_metadata"]
hidden_dim = config["hidden_dim"]
n_layers = config["n_layers"]
bidirectional = config["bidirectional"]

torch.manual_seed(42)

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
print("PyTorch Version : {}".format(torch.__version__))
print(DEVICE)

# Setting up the logging
logging.basicConfig(filename='training_log.txt', level=logging.INFO, format='%(asctime)s - %(message)s')
logging.info("PyTorch Version : {}".format(torch.__version__))
logging.info("Using device: {}".format(DEVICE))


model = LiarModel(vocab_size, embedding_dim, n_filters, filter_sizes, output_dim, dropout, padding_idx, input_dim, input_dim_metadata, hidden_dim, n_layers, bidirectional).to(DEVICE)


# Define the optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.BCEWithLogitsLoss()


# Record the training process
Train_acc = []
Train_loss = []
Train_macro_f1 = []
Train_micro_f1 = []

Val_acc = []
Val_loss = []
Val_macro_f1 = []
Val_micro_f1 = []

def train(num_epochs, model, train_loader, val_loader, optimizer, criterion, model_save):
    epoch_trained = 0
    train_label_all = []
    train_predict_all = []
    val_label_all = []
    val_predict_all = []
    best_valid_loss = float('inf')

    start_time = time.time()
    for epoch in range(num_epochs):
        epoch_trained += 1
        epoch_start_time = time.time()
        # Training
        model.train()
        train_loss = 0.0
        train_accuracy = 0.0
        for statements, label_onehot, label, metadata_text, metadata_number, justification in train_loader:
            statements = statements.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)
            metadata_text = metadata_text.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)
            justification = justification.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(statements, metadata_text, metadata_number, justification)
            loss = criterion(outputs, label_onehot)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, train_predicted = torch.max(outputs, 1)
            train_accuracy += sum(train_predicted == label)
            train_predict_all += train_predicted.tolist()
            train_label_all += label.tolist()
        train_loss /= len(train_loader)
        train_accuracy /= len(train_loader.dataset)
        train_macro_f1 = f1_score(train_label_all, train_predict_all, average='macro')
        train_micro_f1 = f1_score(train_label_all, train_predict_all, average='micro')

        Train_acc.append(train_accuracy.tolist())
        Train_loss.append(train_loss)
        Train_macro_f1.append(train_macro_f1)
        Train_micro_f1.append(train_micro_f1)

        # Validation
        model.eval()
        val_loss = 0.0
        val_accuracy = 0.0
        with torch.no_grad():
            for statements, label_onehot, label, metadata_text, metadata_number, justification in val_loader:
                statements = statements.to(DEVICE)
                label_onehot = label_onehot.to(DEVICE)
                label = label.to(DEVICE)
                metadata_text = metadata_text.to(DEVICE)
                metadata_number = metadata_number.to(DEVICE)
                justification = justification.to(DEVICE)

                val_outputs = model(statements, metadata_text, metadata_number, justification)
                val_loss += criterion(val_outputs, label_onehot).item()
                _, val_predicted = torch.max(val_outputs, 1)
                val_accuracy += sum(val_predicted == label)
                val_predict_all += val_predicted.tolist()
                val_label_all += label.tolist()
        val_loss /= len(val_loader)
        val_accuracy /= len(val_loader.dataset)
        val_macro_f1 = f1_score(val_label_all, val_predict_all, average='macro')
        val_micro_f1 = f1_score(val_label_all, val_predict_all, average='micro')

        Val_acc.append(val_accuracy.tolist())
        Val_loss.append(val_loss)
        Val_macro_f1.append(val_macro_f1)
        Val_micro_f1.append(val_micro_f1)

        if val_loss < best_valid_loss:
            best_valid_loss = val_loss
            torch.save(model.state_dict(), model_save)
            print(f'***** Best Result Updated at Epoch {epoch_trained}, Val Loss: {val_loss:.4f} *****')
            logging.info(f'***** Best Result Updated at Epoch {epoch_trained}, Val Loss: {val_loss:.4f} *****')

        # Print the losses and accuracy
        epoch_end_time = time.time()
        epoch_time = epoch_end_time - epoch_start_time

        log_message = f"Epoch [{epoch+1}/{num_epochs}], Time: {epoch_time:.2f}s, Train Loss: {train_loss:.4f}, Train Acc: {train_accuracy:.4f}, Train F1 Macro: {train_macro_f1:.4f}, Train F1 Micro: {train_micro_f1:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_accuracy:.4f}, Val F1 Macro: {val_macro_f1:.4f}, Val F1 Micro: {val_micro_f1:.4f}"
        print(log_message)
        logging.info(log_message)
        

    end_time = time.time()
    training_time = end_time - start_time
    print(f'Total Training Time: {training_time:.2f}s')
    logging.info(f'Total Training Time: {training_time:.2f}s')



train(num_epochs, model, train_loader(batch_size=batch_size), val_loader(batch_size=batch_size), optimizer, criterion, model_save)