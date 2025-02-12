import torch
from torch_geometric.data import Data
from metrix import node_features, edge_features_tensor, edge_index, edge_labels_tensor
from torch_geometric.nn import GCNConv
import torch.nn.functional as F
import matplotlib.pyplot as plt
import torch.optim.lr_scheduler as lr_scheduler
import os
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_fscore_support

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
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
            torch.nn.Linear(178, 64),  # 注意这里的178应根据实际输入维度调整
            torch.nn.ReLU(),
            torch.nn.Dropout(0.5),  # 增加Dropout层
            torch.nn.Linear(64, num_classes)
        )

    def forward(self, data):
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        x = node_features.unsqueeze(1) 
        x = F.relu(self.conv1(x.float(), edge_index))
        x = F.dropout(x, p=0.5, training=self.training)  # 使用Dropout
        x = F.relu(self.conv2(x, edge_index))
        start, end = edge_index[0], edge_index[1]
        edge_features = torch.cat((x[start], x[end], edge_attr), dim=1)
        return self.edge_mlp(edge_features)

# 假设data是已经准备好的图数据对象
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
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=0.01)  # 增加L2正则化
criterion = torch.nn.CrossEntropyLoss()
scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[50], gamma=0.5) 

def train(data):
    model.train()
    optimizer.zero_grad()
    out = model(data)
    loss = criterion(out, data.y)
    loss.backward()
    optimizer.step()
    scheduler.step()  
    return loss

def test(data):
    model.eval()
    with torch.no_grad():
        out = model(data)
        pred = out.argmax(dim=1)
        correct = pred.eq(data.y).sum().item()
        acc = correct / data.y.size(0)
        precision, recall, f1, _ = precision_recall_fscore_support(data.y.numpy(), pred.numpy(), average='binary')
        return acc, precision, recall, f1, pred

loss_values = []
for epoch in range(500):
    loss = train(train_data)
    loss_values.append(loss)
    if epoch % 25 == 0:
        test_acc, precision, recall, f1, predictions = test(test_data)
        print(f'Epoch {epoch}: Test Accuracy: {test_acc * 100:.2f}%')
        print(f'Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}')
        cm = confusion_matrix(test_data.y.numpy(), predictions.numpy())
        print("Confusion Matrix:\n", cm)

plt.figure(figsize=(10, 5))
plt.plot([x.item() for x in loss_values], label='Training Loss', marker='o')
plt.title("Training Loss Over Epochs")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()