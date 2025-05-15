import json
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
import logging
from model_bio import LiarModel
from data_loader_biometric import test_loader
import csv

def test(model, test_loader, criterion, model_save, input_dim_metadata):
    """
    Load the trained model from 'model_save' and evaluate it on the test set.
    """
    # Load the saved model weights.
    model.load_state_dict(torch.load(model_save))
    model.eval()

    test_label_all = []
    test_predict_all = []
    test_loss = 0.0
    test_accuracy = 0.0

    with torch.no_grad():
        for statements, label_onehot, label, metadata_text, metadata_number, justification in test_loader:
            # Move data to the device.
            statements = statements.to(DEVICE)
            label_onehot = label_onehot.to(DEVICE)
            label = label.to(DEVICE)
            metadata_text = metadata_text.to(DEVICE)
            metadata_number = metadata_number.to(DEVICE)
            justification = justification.to(DEVICE)

            # Forward pass.
            test_outputs = model(statements, metadata_text, metadata_number, justification)
            loss = criterion(test_outputs, label_onehot)
            test_loss += loss.item()

            # Get predictions.
            _, test_predicted = torch.max(test_outputs, 1)
            test_accuracy += sum(test_predicted == label)
            test_predict_all += test_predicted.tolist()
            test_label_all += label.tolist()

    # Average the loss and compute accuracy.
    test_loss /= len(test_loader)
    test_accuracy /= len(test_loader.dataset)
    test_macro_f1 = f1_score(test_label_all, test_predict_all, average='macro')
    test_micro_f1 = f1_score(test_label_all, test_predict_all, average='micro')

    result_str = (
        f"Test Loss: {test_loss:.4f}, "
        f"Test Acc: {test_accuracy:.4f}, "
        f"Test F1 Macro: {test_macro_f1:.4f}, "
        f"Test F1 Micro: {test_micro_f1:.4f}"
    )
    logging.info(result_str)
    print(result_str)
    # Append performance metrics to CSV file
    with open(performance_log_file, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            i+1,
            round(test_loss, 4),
            round(test_accuracy.item(), 4),
            round(test_macro_f1, 4),
            round(test_micro_f1, 4)
        ])




# Load configuration settings from JSON file.
with open('config-biometric.json', 'r') as f:
    config = json.load(f)

for i in range(13):

    # Set up logging to write to a log file.
    logging.basicConfig(filename=f'{i+1}_test_log_bio.txt', 
                        level=logging.INFO, 
                        format='%(asctime)s - %(message)s')

    # Extract configuration parameters.
    # workspace         = config["workspace"]
    model_save        = f"{i+1}_model_bio.pt"
    # model_name = config["model_name"]
    # num_epochs        = config["num_epochs"]
    batch_size        = config["batch_size"]
    # learning_rate     = config["learning_rate"]
    # num_classes       = config["num_classes"]
    padding_idx       = config["padding_idx"]
    # metadata_each_dim = config["metadata_each_dim"]

    vocab_size        = config["vocab_size"]
    embedding_dim     = config["embedding_dim"]
    n_filters         = config["n_filters"]
    filter_sizes      = config["filter_sizes"]
    output_dim        = config["output_dim"]
    dropout           = config["dropout"]
    input_dim         = config["input_dim"]
    # input_dim_metadata= config["input_dim_metadata"]
    input_dim_metadata = 1
    # trigram_embedding_dim = config["trigram_embedding_dim"]
    hidden_dim        = config["hidden_dim"]
    n_layers          = config["n_layers"]
    bidirectional     = config["bidirectional"]

    performance_log_file = "1-13_test_performance_log.csv"
    # Define CSV filename
    if i == 0:
        with open(performance_log_file, mode='w', newline='') as f:
            writer = csv.writer(f)
            # Write header row
            writer.writerow([
                "Feature#",
                "Test Loss",
                "Test Accuracy",
                "Test F1 Macro",
                "Test F1 Micro"
            ])

    # Set up device.
    DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    logging.info("PyTorch Version: {}".format(torch.__version__))
    logging.info("Using device: {}".format(DEVICE))
    print("PyTorch Version: {}".format(torch.__version__))
    print("Using device: {}".format(DEVICE))

    # Initialize the model with parameters from the JSON config.
    model = LiarModel(
        vocab_size, 
        embedding_dim, 
        n_filters, 
        filter_sizes, 
        output_dim, 
        dropout, 
        padding_idx, 
        input_dim, 
        input_dim_metadata, 
        hidden_dim, 
        n_layers, 
        bidirectional
    ).to(DEVICE)

    # Define the loss function.
    criterion = nn.BCEWithLogitsLoss()



    # Run the test function.
    test(model, test_loader(batch_size=batch_size, input_dim_metadata=i+1), criterion, model_save, input_dim_metadata)

