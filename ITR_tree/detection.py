import torch
from build_ITR_tree import build_ITR_tree
from train import generate_square_subsequent_mask
import pickle
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import csv
from pymongo import MongoClient
import pymongo
from CutomDataset import CustomDataset_det
import numpy as np
import math
import glob
from data_process import process_entry
from build_tree_and_vocabulary import build_tree_and_vocabulary
from build_WordEmbedding import build_WordEmbedding
from traverse_tree import traverse_tree 

# 连接数据库，读取数据，
client = MongoClient('mongodb://b515:sqwUiJGHYQTikv6z@10.12.46.33:27018/?authMechanism=DEFAULT')
dbtest = client["geth"]
collection = dbtest.get_collection("test_attack_processed")
data = collection.find()

# with open('../data/train_tx_score/random_all.csv', 'w',newline='') as csv_file:
#     for result in data:
#         csv_writer = csv.writer(csv_file)
#         csv_writer.writerow([result['tx_hash']])

file_i = 1366
vocabulary_file = "../vocabulary/random_new_1.pkl"

with open(vocabulary_file,"rb") as file:   
        vocabulary = pickle.load(file)
encoder_model = torch.load('../model/random_new_1.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_file_train = []
for i in range (file_i):
    embedding_file_train.append(f"../tmp/test_new_1/test_embedding/tree_node_list{i}.pth")

custom_train = CustomDataset_det(embedding_file_train,vocabulary)
train_data_loader = DataLoader(custom_train, batch_size=1, shuffle=False,pin_memory=True)

encoder_model.eval()  # 开启评估模式
loss_function = nn.CrossEntropyLoss()
src_mask = generate_square_subsequent_mask(64).to(device)

# 指定要写入的CSV文件路径
csv_file_path = '../data/train_new_1/test_attack/tx_score.csv'
# 打开CSV文件并写入数据
with open(csv_file_path, 'w',newline='') as csv_file:
    softmax = nn.Softmax(dim=-1)
    with torch.no_grad():
        i= 0
        #log_likelihoods = []
        #for row in reader:
        for batch ,one_hot ,src_key_mask in train_data_loader:
            batch, one_hot ,src_key_mask= batch.to(device), one_hot.to(device),src_key_mask.to(device)
        
            outputs = encoder_model(batch,src_mask,src_key_mask)
            loss = loss_function(outputs.view(-1,vocabulary.index), one_hot.reshape(-1))
            
            
            # output = softmax(outputs).log()
            # log_likelihood = output.sum()
            # print(log_likelihood.item())

            # exit()
         
            #log_likelihoods.append(log_likelihood.item())
            #loss = loss_function(outputs.view(-1,vocabulary.index), one_hot.reshape(-1))
            #self.softmax = nn.LogSoftmax(dim=2)  # 在此处使用对数softmax
            #log_likelihoods = model(sentence).sum()  # 计算对数似然值，此处假设句子的对数似然值是各个词汇概率的总和
            #row.append([log_likelihood.item()])
            result = collection.find_one(skip=i, sort=[('_id', pymongo.ASCENDING)])
            i +=1
            csv_writer = csv.writer(csv_file)

            # # 使用写入器对象写入数据
            csv_writer.writerow([result['tx_hash'],loss.item()])
# alpha = 0.1  # 举例，标记10%最不正常的交易
# threshold = np.percentile(log_likelihoods, alpha * 100)
# abnormal_transactions = [i for i, ll in enumerate(log_likelihoods) if ll < threshold]
# print(abnormal_transactions)

