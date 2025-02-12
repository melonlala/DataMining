import random
from load_data_new import new_batches
import random
import numpy as np
import torch

def process_batches(new_batches):
    num_0 = 0
    num_1 = 0
    combined_data = [[] for _ in range(8)]
    combined_data.insert(4, [[], []])  # combined_data[4] has two empty lists

    for batch in new_batches:
        labels = batch[6]
        for i in range(512):
            if (labels[i] == 0 and num_0 < 30 * 512) or (labels[i] == 1 and num_1 < 30 * 512):
                for j in [0, 1, 2, 3, 5, 6, 7]:
                    combined_data[j].append(batch[j][i])
                combined_data[4][0].append(batch[4][0][i])
                combined_data[4][1].append(batch[4][1][i])
                if labels[i] == 0:
                    num_0 += 1
                else:
                    num_1 += 1

    # Shuffle data
    combined_indices = list(range(num_0 + num_1))
    random.shuffle(combined_indices)
    shuffled_data = [[] for _ in range(8)]
    shuffled_data[4] = [[], []]  # Initialize nested lists for index 4
    for index in combined_indices:
        for j in [0, 1, 2, 3, 5, 6, 7]:
            shuffled_data[j].append(combined_data[j][index])
        shuffled_data[4][0].append(combined_data[4][0][index])
        shuffled_data[4][1].append(combined_data[4][1][index])
    
    # Split data into batches
    num_batches = (num_0 + num_1) // 512
    balanced_batches = [[shuffled_data[j][i * 512:(i + 1) * 512] for j in range(8)] for i in range(num_batches)]
    balanced_batches_4_0 = [shuffled_data[4][0][i * 512:(i + 1) * 512] for i in range(num_batches)]
    balanced_batches_4_1 = [shuffled_data[4][1][i * 512:(i + 1) * 512] for i in range(num_batches)]
    
    # Combine balanced_batches_4_0 and balanced_batches_4_1 into balanced_batches[4]
    for i in range(num_batches):
        balanced_batches[i][4] = [balanced_batches_4_0[i], balanced_batches_4_1[i]]
    
    return balanced_batches
balance_batches = process_batches(new_batches)

def get_numpy_array(x):
    if isinstance(x, torch.Tensor):
        return x.numpy()
    elif isinstance(x, np.ndarray):
        return x
    else:
        return np.array(x)

unique_nodes = np.unique(np.concatenate([np.concatenate([get_numpy_array(b[0]), get_numpy_array(b[1])]) for b in balance_batches]))

# 创建节点ID到连续索引的映射
node_to_index = {int(node): idx for idx, node in enumerate(unique_nodes)}

# 替换原始batch中的节点ID，并将更新后的ID转换回Tensor格式
new_balance_batches = []
for batch in balance_batches:
    start_nodes = torch.tensor([node_to_index[int(node.item())] for node in batch[0]], dtype=torch.int64)
    end_nodes = torch.tensor([node_to_index[int(node.item())] for node in batch[1]], dtype=torch.int64)
    new_balance_batches.append((start_nodes, end_nodes))

for i in  range(60):
    for j in range(512):
        balance_batches[i][0][j] = new_balance_batches[i][0][j]
        balance_batches[i][1][j] = new_balance_batches[i][1][j]
        

# print(len(balance_batches))
# print(len(balance_batches[0]))
i = 50
j = 100
print("Batch Source IDs:", balance_batches[i][0][j]) 
print("Batch Destination IDs:", balance_batches[i][1][j])  
# print("Batch Words (Content):", balance_batches[0][2])  
print("Batch Users:", balance_batches[i][3][j])  
print("Batch Subreddits for Source:", balance_batches[i][4][0][j])  
print("Batch Subreddits for Destination:", balance_batches[i][4][1][j])  
print("Batch Lengths:", balance_batches[i][5][j]) 
print("Batch Labels:", balance_batches[i][6][j])  
print("Batch Post Times:", balance_batches[i][7][j])  