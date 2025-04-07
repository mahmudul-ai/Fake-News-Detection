import matplotlib.pyplot as plt
import re

# Function to read the log data from a file
def read_log_file(file_path):
    with open(file_path, 'r') as file:
        return file.read()

# Sample file path (replace with the actual path to your log file)
file_path = 'training_log_bio.txt'  # Replace with your actual file path

# Read the log data from the file
log_data = read_log_file(file_path)

# Regular expression to extract relevant metrics
pattern = r"Epoch \[(\d+)/30\],.*Train Loss: ([\d\.]+), Train Acc: ([\d\.]+),.*Val Loss: ([\d\.]+), Val Acc: ([\d\.]+)"

epochs = []
train_losses = []
train_acc = []
val_losses = []
val_acc = []

# Extracting data from the log text
for line in log_data.split('\n'):
    match = re.search(pattern, line)
    if match:
        epoch = int(match.group(1))
        train_loss = float(match.group(2))
        train_accuracy = float(match.group(3))
        val_loss = float(match.group(4))
        val_accuracy = float(match.group(5))

        epochs.append(epoch)
        train_losses.append(train_loss)
        train_acc.append(train_accuracy)
        val_losses.append(val_loss)
        val_acc.append(val_accuracy)

# Plotting the results
plt.figure(figsize=(14, 6))

# Plotting Train and Validation Loss
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, label='Train Loss', color='blue', marker='o')
plt.plot(epochs, val_losses, label='Val Loss', color='red', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Train and Validation Loss')
plt.legend()

# Plotting Train and Validation Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs, train_acc, label='Train Accuracy', color='green', marker='o')
plt.plot(epochs, val_acc, label='Val Accuracy', color='orange', marker='x')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Train and Validation Accuracy')
plt.legend()

plt.tight_layout()
plt.show()
