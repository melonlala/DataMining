import constants
from embeddings import Embeddings
import torch
import random
import argparse
import pickle
import torch.nn as nn
import numpy as np
import os

print("Checking file at:", constants.POST_INFO)
def load_data(batch_size, max_len):
    print("Loading train/test data...")
    thread_to_sub = {}
    comm = {}
    with open(constants.POST_INFO) as fp:
        count = 0
        tol = 0
        for line in fp:
            tol += 1
            info = line.split()
            source_sub = info[0]
            target_sub = info[1]
            source_post = info[2].split("T")[0].strip()
            target_post = info[6].split("T")[0].strip()
            if source_post in thread_to_sub or target_post in thread_to_sub:
                # print("DUPLICATE THREAD:", source_post, target_post)
                count += 1
            thread_to_sub[source_post] = source_sub
            thread_to_sub[target_post] = target_sub
            if comm.get(source_sub) is None:
                comm[source_sub] = [source_post]
            else:
                comm[source_sub].append(source_post)
            if comm.get(target_sub) is None:
                comm[target_sub] = [target_post]
            else:
                comm[target_sub].append(target_post)
        print("Total duplicate threads:", count)
        print("Total threads:", tol)

    label_map = {}
    source_to_dest = {}
    with open(constants.LABEL_INFO) as fp:
        for line in fp:
            info = line.split("\t")
            source = info[0].split(",")[0].split("\'")[1]
            dest = info[0].split(",")[1].split("\'")[1]
            label_map[source] = 1 if info[1].strip() == "burst" else 0
            try:
                assert source in thread_to_sub
                assert dest in thread_to_sub
            except AssertionError:
                continue
            source_to_dest[source] = dest

    with open(constants.SUBREDDIT_IDS) as fp:
        sub_id_map = {sub:i for i, sub in enumerate(fp.readline().split())}

    with open(constants.USER_IDS) as fp:
        user_id_map = {user:i for i, user in enumerate(fp.readline().split())}

    with open(constants.PREPROCESSED_DATA) as fp:
        words, users, dests, subreddits, lengths, labels, post_times, ids = [], [], [], [], [], [], [], []
        post_info = {}
        for i, line in enumerate(fp):
            info = line.split("\t")
            post_name = info[1]
            post_time = info[3]
            if post_name in source_to_dest :
                dest = source_to_dest[post_name]
                dests.append(dest)
                try:
                    assert info[0] == thread_to_sub[post_name]
                except AssertionError:
                    print("sub mismatch!")
                    continue
                source_sub = info[0]
                dest_sub = thread_to_sub[dest]



                title_words = info[-2].split(":")[1].strip().split(",")
                title_words = title_words[:min(len(title_words), constants.MAX_LEN)]
                if len(title_words) == 0 or title_words[0] == '':
                    continue
                words.append(list(map(int, title_words)))

                body_words = info[-1].split(":")[1].strip().split(",")
                body_words = body_words[:min(len(body_words), constants.MAX_LEN-len(title_words))]
                if not (len(body_words) == 0 or body_words[0] == ''):
                    words[-1].extend(list(map(int, body_words)))

                words[-1] = [constants.VOCAB_SIZE+1 if w==-1 else w for w in words[-1]]
                if not info[0] in sub_id_map:
                    source_sub_id = constants.NUM_SUBREDDITS
                else:
                    source_sub_id = sub_id_map[info[0]]
                
                if not dest_sub in sub_id_map:
                    dest_sub_id = constants.NUM_SUBREDDITS
                else:
                    dest_sub_id = sub_id_map[dest_sub]
            
                subreddits.append([source_sub_id, dest_sub_id])
                users.append([constants.NUM_USERS if not info[2] in user_id_map else user_id_map[info[2]]])
                ids.append(post_name)
                post_times.append(post_time)
                lengths.append(len(words[-1])+3)
                labels.append(label_map[post_name])
                post_info[post_name] = [post_name, post_time, [source_sub, dest_sub]]
        
    batches = []
    np.random.seed(0)
    for count, i in enumerate(np.random.permutation(len(words))):
        if count % batch_size == 0:
            batch_words = np.ones((max_len, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
            batch_users = np.ones((1, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
            batch_subs = np.ones((2, batch_size), dtype=np.int64) * constants.VOCAB_SIZE
            batch_lengths = []
            batch_labels = []
            batch_ids = []
            batch_times = []
            batch_dests = []
        length = min(max_len, len(words[i]))
        batch_words[:length, count % batch_size] = words[i][:length]
        batch_users[:, count % batch_size] = users[i]
        batch_subs[:, count % batch_size] = subreddits[i]
        batch_dests.append(dests[i])
        batch_lengths.append(length)
        batch_labels.append(labels[i])
        batch_ids.append(ids[i])
        batch_times.append(post_times[i])
        #print(f"Shape of batch_subs before conversion: {np.array(batch_subs).shape}")
        if count % batch_size == batch_size - 1:
            order = np.flip(np.argsort(batch_lengths), axis=0)
            batches.append((
                np.array(batch_ids)[order],
                np.array(batch_dests)[order],
                torch.LongTensor(batch_words[:, order]), 
                torch.LongTensor(batch_users[:, order]), 
                torch.LongTensor(batch_subs[:, order]), 
                np.array(batch_lengths)[order],
                torch.LongTensor(np.array(batch_labels)[order]),
                np.array(batch_times)[order]
            ))


    return batches

batches = load_data(constants.BATCH_SIZE, constants.MAX_LEN)
print("len(batches)",len(batches))
new_batches = []
import torch

id_to_index = {}
current_index = 0
for batch in batches:
    all_ids = list(batch[0]) + list(batch[1])  
    for post_id in all_ids:
        if post_id not in id_to_index:
            id_to_index[post_id] = current_index
            current_index += 1

for i in range(len(batches)):
    batch_ids, batch_dests, batch_words, batch_users, batch_subs, batch_lengths, batch_labels, batch_times = batches[i]
    index_list_ids = [id_to_index[post_id] for post_id in batch_ids]
    index_list_dests = [id_to_index[post_id] for post_id in batch_dests]
    index_tensor_ids = torch.LongTensor(index_list_ids)
    index_tensor_dests = torch.LongTensor(index_list_dests)
    batch_users = batch_users.transpose(0, 1)
    batch_words = batch_words.transpose(0, 1)
    new_batch = (index_tensor_ids, index_tensor_dests, batch_words, batch_users, batch_subs, batch_lengths, batch_labels, batch_times)
    new_batches.append(new_batch)

i = 100
j = 100
# print("Batch Source IDs:", new_batches[0][0].shape) 
# print("Batch Words (Content):", new_batches[0][2].shape)
# print("Batch Source IDs:", new_batches[i][0][j]) 
# print("Batch Destination IDs:", new_batches[i][1][j])  
# print("Batch Words (Content):", batches[0][2])  
# print("Batch Users:", batches[i][3][j])  
# print("Batch Subreddits for Source:", batches[i][4][0][j])  
# print("Batch Subreddits for Destination:", batches[i][4][1][j])  
# print("Batch Lengths:", batches[i][5][j]) 
# print("Batch Labels:", batches[0][6].shape)  
# print("Batch Post Times:", batches[i][7][j])  