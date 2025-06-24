import torch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

import torch.nn.functional as F
import time
import os
from scipy.special import rel_entr

import torch
import torch.nn as nn
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.tanh = nn.Tanh()

    def forward(self, lstm_output):
        query = self.linear(lstm_output[:, -1]).unsqueeze(2)
        keys = lstm_output
       
        energy = torch.bmm(keys, query).squeeze(2)
        
        attention_weights = F.softmax(energy, dim=1).unsqueeze(1)
        context = torch.bmm(attention_weights, lstm_output).squeeze(1)
        return context


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.attention = Attention(hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        context = self.attention(out)
        out = self.fc(context)
        out = F.softmax(out, dim=1)
        return out


input_data = np.load("data/graph_predict/all_input_hotel_reserv.npy")
target_data = np.load("data/graph_predict/all_target_hotel_reserv.npy")

target_data = target_data[:, 0, :]

epsilon = 1e-8
target_data += epsilon
target_data = target_data / target_data.sum(axis=1, keepdims=True)

input_size = input_data.shape[2]
output_size = target_data.shape[1]

hidden_size = 64
num_layers = 2




model = LSTMModel(input_size, hidden_size, num_layers, output_size)


criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)




x_train, x_test, y_train, y_test = train_test_split(
    input_data, target_data, test_size=0.1
)

x_train = torch.from_numpy(x_train.astype(np.float32))
y_train = torch.from_numpy(y_train.astype(np.float32))
x_test = torch.from_numpy(x_test.astype(np.float32))
y_test = torch.from_numpy(y_test.astype(np.float32))

dataset = torch.utils.data.TensorDataset(x_train, y_train)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=100, shuffle=True)


def train():
    num_epochs = 30
    for epoch in range(num_epochs):
        for inputs, targets in dataloader:

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

    print("outputs.shape:", outputs.shape)
    print("targets.shape:", targets.shape)
    print("outputs[0]:", outputs[0])
    print("targets[0]:", targets[0])
    os.system("mkdir -p models")
    torch.save(model.state_dict(), "models/graph_predict.pt")


# def test():
#     model.load_state_dict(torch.load("models/graph_predict_Alibaba.pt"))
#     test_output = model(x_test)
#     loss = criterion(test_output, y_test)
#     print(test_output)
#     print(f"Loss: {float(loss)}")
    
def test():
    model.load_state_dict(torch.load("models/graph_predict_Alibaba.pt"))
    model.eval()
    with torch.no_grad():
        test_output = model(x_test)  # log_softmax output
        loss = criterion(test_output, y_test)
        
        print(f"Loss: {float(loss):.4f}")

        # log_softmax → softmax로 복원
        # probs = test_output.exp()  # shape: [num_samples, output_dim]
        probs = test_output
        # 예측 클래스와 실제 클래스 인덱스 (argmax 기준)
        pred_indices = torch.argmax(probs, dim=1)
        true_indices = torch.argmax(y_test, dim=1)

        # 정확도 계산
        correct = (pred_indices == true_indices).sum().item()
        total = y_test.size(0)
        accuracy = correct / total * 100

        print(f"Accuracy: {accuracy:.2f}%")
        
        # 첫 샘플의 예측 분포와 실제 분포 비교 출력
        print("\n[Sample 0 Prediction vs Target]")
        print(f"Predicted index: {pred_indices[0].item()}, True index: {true_indices[0].item()}")
        print(f"Predicted prob (top): {probs[0][pred_indices[0]].item():.4f}")
        print(f"Target prob (top): {y_test[0][true_indices[0]].item():.4f}")
        print(f"Predicted distribution (first 10): {probs[0][:10].numpy()}")
        print(f"Target distribution (first 10):    {y_test[0][:10].numpy()}")
    


def evaluate():
    model.load_state_dict(torch.load("models/graph_predict_Alibaba.pt"))
    bin_size=1
    model.eval()
    with torch.no_grad():
        preds = model(x_test)

    kl_divs = []
    for i in range(len(y_test)):
        p = preds[i]
        q = y_test[i]
        p = torch.clamp(p, min=1e-8)
        q = torch.clamp(q, min=1e-8)
        kl = F.kl_div(p.log(), q, reduction='sum').item()
        kl_divs.append(kl)

    num_bins = len(kl_divs) // bin_size
    binned_kl = []
    for i in range(num_bins):
        start = i * bin_size
        end = start + bin_size
        avg_kl = np.mean(kl_divs[start:end])
        binned_kl.append(avg_kl)

    # Plot
    plt.figure(figsize=(10, 5))
    plt.plot(range(num_bins), binned_kl, marker='o', linestyle='-', color='teal')
    plt.title(f"KL Divergence (every samples)", fontsize=20)
    plt.xlabel("Sample Index", fontsize=20)
    plt.ylabel("Average KL Divergence", fontsize=20)
    plt.grid(True, linestyle='--', alpha=0.4)
    plt.tight_layout()

    plt.savefig("figures/evaluation_sample_detailed.png")
    plt.show()

    print(f"Average KL Divergence: {np.mean(kl_divs):.6f}")

        


    
if __name__ == "__main__":
    import sys

    module = sys.argv[1]
    if module == "train":
        start = time.perf_counter()
        train()
        end = time.perf_counter()
        print(f"Train time: {end - start} seconds")
    if module == "test":
        test()
    if module == "evaluate":
        evaluate()
