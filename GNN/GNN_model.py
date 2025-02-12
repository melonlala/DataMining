import torch
from torch_geometric.data import Data
from metrix import node_features, edge_features_tensor,edge_index,edge_labels_tensor
from torch_geometric.nn import GCNConv, global_add_pool, MessagePassing
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
os.environ['KMP_DUPLICATE_LIB_OK']='True'


# 构建PyG的图数据对象
data = Data(
    x=node_features.float(),
    edge_index=edge_index,
    edge_attr=edge_features_tensor.float(),
    y=edge_labels_tensor
)

class EdgeClassifier(torch.nn.Module):
    def __init__(self, num_node_features, num_edge_features, num_classes):
        super(EdgeClassifier, self).__init__()
        self.conv1 = GCNConv(num_node_features, 128)
        self.conv2 = GCNConv(128, 64)
        self.edge_mlp = torch.nn.Sequential(
            torch.nn.Linear(640, 64),  
            torch.nn.ReLU(),
            torch.nn.Linear(64, num_classes) 
        )

    
    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = node_features.unsqueeze(1) 
        x = F.relu(self.conv1(x.float(), edge_index))
        x = F.dropout(x, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        start, end = edge_index[0], edge_index[1]
        edge_features = torch.cat((x[start], x[end], edge_attr), dim=1)
        #print("Edge features shape:", edge_features.shape)
        return self.edge_mlp(edge_features)

    
num_edges = data.edge_index.size(1)
indices = np.random.permutation(num_edges)
train_size = int(0.8 * num_edges)
train_indices = indices[:train_size]
test_indices = indices[train_size:]

train_data = Data(
    x=data.x,
    edge_index=data.edge_index[:, train_indices],
    edge_attr=data.edge_attr[train_indices],
    y=data.y[train_indices]
)

test_data = Data(
    x=data.x,
    edge_index=data.edge_index[:, test_indices],
    edge_attr=data.edge_attr[test_indices],
    y=data.y[test_indices]
)

model = EdgeClassifier(num_node_features=data.num_node_features, num_edge_features=640, num_classes=2)
optimizer = torch.optim.Adam(model.parameters(), lr=0.005)
criterion = torch.nn.CrossEntropyLoss()
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[150], gamma=2.0)

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    return loss

def test(data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = pred.eq(data.y).sum().item()
        return correct / data.y.size(0)

loss_values = []
for epoch in range(300):
    loss = train(train_data)
    loss_values.append(loss)
    if epoch % 20 == 0:  # Check if the current epoch number is divisible by 20
        test_acc = test(test_data)
        print(f'Epoch {epoch}: Test Accuracy: {test_acc * 100:.2f}%')

plt.figure(figsize=(10, 5))
plt.plot([x.item() for x in loss_values], label='Training Loss', marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()