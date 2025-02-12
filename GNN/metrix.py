from torch_geometric.data import Data
from proceed_data import balance_batches
import torch

all_start_nodes = []
all_end_nodes = []
node_features = torch.full((59256,), -1, dtype=torch.long)
all_edge_features = []
max_length = max(max(len(feature) for feature in batch[2]) for batch in balance_batches)
all_edge_labels = []
count = 0
maxnode = 0
for batch in balance_batches:
    start_nodes = batch[0]
    end_nodes = batch[1]
    start_communities = batch[4][0]
    end_communities = batch[4][1]
    all_start_nodes.extend(start_nodes)
    all_end_nodes.extend(end_nodes)
    edge_labels = batch[6]
    all_edge_labels.extend(edge_labels)
    for j in range(len(start_nodes)):
        node_features[start_nodes[j]] = start_communities[j]
        node_features[end_nodes[j]] = end_communities[j]
        maxnode = max(start_nodes[j],end_nodes[j],maxnode)
    for features in batch[2]:
        count +=1
        padding_length = max_length - features.size(0)
        padding = torch.zeros(padding_length, dtype=features.dtype)
        padded_features = torch.cat((features, padding), dim=0)
        all_edge_features.append(padded_features)

all_start_nodes = torch.tensor(all_start_nodes, dtype=torch.long)
all_end_nodes = torch.tensor(all_end_nodes, dtype=torch.long)
edge_index = torch.stack([all_start_nodes, all_end_nodes], dim=0)
edge_labels_tensor = torch.tensor(all_edge_labels, dtype=torch.long)
edge_features_tensor = torch.stack(all_edge_features, dim=0)

print("Edge Index Matrix:")
print(edge_index.shape)
print("Node Features Matrix:")
print(node_features.shape)
print("Edge Features Matrix:")
print(edge_features_tensor.shape)
print("Edge Labels Matrix:")
print(edge_index.shape)
# print("Edge Index Matrix:")
# print(edge_index)
# print("Node Features Matrix:")
# print(node_features)
# print("Edge Features Matrix:")
# print(edge_features_tensor)
# print("Edge Labels Matrix:")
# print(edge_labels_tensor)
# print(edge_index)