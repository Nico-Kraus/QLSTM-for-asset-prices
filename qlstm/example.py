import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from lstm_tagger import LSTMTagger

from matplotlib import pyplot as plt

# size of the data is n
# period of the sequence is p
# k is an additional factor for each number in the series
# lookback is the length of the sequence for each training step
n = 100
p = 4
k = 0
lookback = 10

# more complicated data
# data = [[k + i % p / 10 + i % p] for i in range(n * p)]

# simple data
data = [[i % p] for i in range(n * p)]

train_data_x = []
train_data_y = []
for i in range(len(data) - lookback):
    train_data_x.append(data[i : i + lookback])
    train_data_y.append(data[i + lookback : i + lookback + 1])

print(train_data_x)
# print(train_data_y)

train_size = len(train_data_x)
# print(train_size)

# model params
# n_qubits = 0 means we use a classical LSTM
# backend only for QLSTM relevant
# for batch size = 1 the results are great, batch size = 2 already does not work
hidden_dim = 4
num_layers = 2
n_qubits = 0
n_epochs = 50
batch_size = 2
backend = "default.qubit"
lr = 0.001

print(f"LSTM output size: {hidden_dim}")
print(f"Number of qubits: {n_qubits}")
print(f"Training epochs:  {n_epochs}")
print(f"Batch size: {batch_size}")

model = LSTMTagger(
    input_size=1,
    hidden_dim=hidden_dim,
    num_layers=num_layers,
    output_size=1,
    n_qubits=n_qubits,
    backend=backend,
)

# calculate number of batches, depending on the lenght of train data and the batch size
num_batches = int(train_size / batch_size) - 1

loss_function = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)

history = {"loss": [], "acc": []}
for epoch in range(n_epochs):

    for batch in range(num_batches):

        x = train_data_x[batch * batch_size : (batch + 1) * batch_size]
        y = train_data_y[batch * batch_size : (batch + 1) * batch_size]

        losses = []
        preds = []
        targets = []

        # Step 1. Remember that Pytorch accumulates gradients.
        # We need to clear them out before each instance
        model.zero_grad()

        # Step 2. Get our inputs ready for the network, that is, turn them into
        # Tensors
        input = torch.tensor(x, dtype=torch.float32)
        labels = torch.tensor(y, dtype=torch.float32)

        # Step 3. Run our forward pass.
        prediction = model(input)

        # Step 4. Compute the loss, gradients, and update the parameters by
        #  calling optimizer.step()
        loss = loss_function(prediction, labels)
        loss.backward()
        optimizer.step()
        losses.append(float(loss))

    probs = torch.tensor(list(prediction))
    preds.append(probs)
    targets.append(labels)

    avg_loss = np.mean(losses)
    history["loss"].append(avg_loss)

    preds = torch.cat(preds)
    targets = torch.cat(targets)
    # a value is correct if it is closer than 0.1 to the real value
    corrects = torch.tensor(
        [np.isclose(preds[i], targets[i], atol=0.1) for i in range(len(targets))]
    )
    accuracy = corrects.sum().float() / float(targets.size(0))
    history["acc"].append(accuracy)

    print(f"Epoch {epoch+1} / {n_epochs}: Loss = {avg_loss:.3f} Acc = {accuracy:.2f}")

# See what the scores are after training
with torch.no_grad():

    tag_scores = model(torch.tensor(train_data_x, dtype=torch.float32))

    pred = tag_scores.flatten().tolist()

    # print(f"Input:  {train_data_x}")
    # print(f"Output:    {train_data_y}")
    print(f"Predicted: {pred}")

lstm_choice = "classical" if n_qubits == 0 else "quantum"

fig, ax1 = plt.subplots()

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Loss")
ax1.plot(history["loss"], label=f"{lstm_choice} LSTM Loss")

ax2 = ax1.twinx()
ax2.set_ylabel("Accuracy")
ax2.plot(history["acc"], label=f"{lstm_choice} LSTM Accuracy", color="tab:red")

plt.title("Part-of-Speech Tagger Training")
plt.ylim(0.0, 1.5)
plt.legend(loc="upper right")

plt.savefig(f"training_{lstm_choice}.png")
plt.show()
