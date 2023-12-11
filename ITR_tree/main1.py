# 用于另一实验的embedding
from pymongo import MongoClient
import torch
from build_WordEmbedding import build_WordEmbedding
from build_tree_and_vocabulary import build_tree_and_vocabulary
from build_vocabulary import Vocabulary
from data_process import process_entry
import time
from train import train_and_eval
import pickle
# from traverse_tree import traverse_tree 
import datetime
from build_ITR_tree import build_ITR_tree
from tokenize_text import tokenize_tree
import numpy as np
import psutil
import os
import csv

# 连接数据库，读取数据，
client = MongoClient('mongodb://b515:sqwUiJGHYQTikv6z@10.12.46.33:27018/?authMechanism=DEFAULT')
dbtest = client["geth"]
collection = dbtest.get_collection("1k_normal_1k_anomoly")


data = collection.find()
data_onehot = collection.find()

# 处理交易时要用的一些标量
# tree_node_list = [] # 把node放入list中
tensor_list = {}    # tensor词典，格式是{'tensor_x':embedding}
vocabulary = Vocabulary()
file_i = 0  # file_i是将数据embedding后保存至第几个文件
file_i_onehot = 0
data_i = 0  # data_i是记录读取的交易个数
data_i_onehot = 0
tensor_i = 0 #每隔几个交易保存一次文件
tensor_i_onehot = 0
#tx_token_max = 0 #记录最大的交易token数量
one_hot = [] # 每个token的one hot编码
one_hot_list = {} #

# 增加开始训练的时间
current_datetime = datetime.datetime.now()
formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
start_time = time.time()
file_name = "../log/log0.txt"
with open(file_name, "a") as file:
                file.write('start date:    '+ formatted_time + "\n")
                
# #检测交易时候增加的内容
# with open("../vocabulary/random_new.pkl","rb") as file:   
#         vocabulary = pickle.load(file)

# 开始处理每一个交易，输出一个交易嵌入，[token总数目,64]
for idx,entry in enumerate(data):
    
    # 空的embdding，用于等下矩阵拼接
    tx_embedding = torch.zeros(0,64)
    # 交易个数+1
    data_i +=1

    # 预处理交易，把交易变成树，生成词汇表
    # 1 处理每一条记录
    Seqsstate_1, Seqslog_1, Seqscall_1 = process_entry(entry)

    # 如果检测的话，也需要在这里生成embdding，使用的是建立好的词汇表
    #root_node = build_ITR_tree(Seqsstate_1, Seqslog_1, Seqscall_1)
    #tokenize_tree(root_node)

    # 2  生成树结点，树从根结点开始，和词汇表
    root_node, vocabulary = build_tree_and_vocabulary(Seqsstate_1, Seqslog_1, Seqscall_1,vocabulary) # node.children [(结点,call),(),()] node.data ['token','token']
    d_model = 64

    # 3 为整棵树建立embedding矩阵
    build_WordEmbedding(root_node, vocabulary, vocabulary.index, d_model)

    # 先放入根结点的embeddding
    tx_embedding = torch.cat((tx_embedding,root_node.embedding),dim=0)
    # 遍历树的所有结点，将结点下面的矩阵全部拼接在一起
    while root_node.children:
        # 再判断根结点下面有没有call
        if root_node.children[0][0].tag == 'call':
            for i in  range(len(root_node.children)):
                if i ==0:
                    continue
                else:
                    tx_embedding = torch.cat((tx_embedding,root_node.children[i][0].embedding),dim=0)

            root_node = root_node.children[0][0]
        else:
            for i in range(len(root_node.children)):
                tx_embedding = torch.cat((tx_embedding,root_node.children[i][0].embedding),dim=0)
            break
    
    if tx_embedding.shape[0]<64:
        padding = 64 - tx_embedding.shape[0]
        padding_tensor = torch.zeros(padding, 64)
        tx_embedding = torch.cat((tx_embedding, padding_tensor), dim=0)
    elif tx_embedding.shape[0]>64:
        tx_embedding = tx_embedding[:64, :]

    # 将交易嵌入保存在tensor_list
    tensor_list[f"tensor_{tensor_i}"] = tx_embedding
    tensor_i += 1

    # 每隔几个交易将embedding保存到文件
    if data_i % 32 == 0 :

        file_path = f"../../finetune/data/data_process/embedding/tree_node_list{file_i}.pth"
        with open(file_path, "w") as file:
            torch.save(tensor_list,file_path)

        tensor_list = {}
        file_i += 1
        tensor_i = 0

# 为交易生成one hot编码
for idx,entry in enumerate(data_onehot):
    one_hot = []
    data_i_onehot +=1
    
    Seqsstate_1, Seqslog_1, Seqscall_1 = process_entry(entry)
    root_node = build_ITR_tree(Seqsstate_1, Seqslog_1, Seqscall_1)
    tokenize_tree(root_node)
    
    for token in root_node.data:
        one_hot.append(vocabulary.get_index(token))

    while root_node.children:
        # 再判断根结点下面有没有call
        if root_node.children[0][0].tag == 'call':
            for i in  range(len(root_node.children)):
                if i ==0:
                    continue
                else:
                    for token in root_node.children[i][0].data:
                        one_hot.append(vocabulary.get_index(token))
            root_node = root_node.children[0][0]
        else:
            for i in range(len(root_node.children)):
                for token in root_node.children[i][0].data:
                        one_hot.append(vocabulary.get_index(token))
            break
    
    # 将one_hot如果小于个，就增加至个，如果大于个，就截断
    while len(one_hot)<65:
        one_hot.append(0)
    if len(one_hot)>65:
        one_hot = one_hot[:65]
    one_hot = one_hot[1:65]
    one_hot = torch.tensor(one_hot)
    one_hot_list[f"tensor_{tensor_i_onehot}"] = one_hot
    tensor_i_onehot += 1
    if data_i_onehot % 32 == 0 :
        file_path = f"../../finetune/data/data_process/one_hot/one_hot_list{file_i_onehot}.pth"
        with open(file_path, "w") as file:
            torch.save(one_hot_list,file_path)
        file_i_onehot += 1
        tensor_i_onehot = 0
        one_hot_list ={}

    # 将词汇表保存到文件
vocabulary_file = "../../finetune/data/data_process/vocabulary/1k_normal_1k_anomoly.pkl"
with open(vocabulary_file, "wb") as file:
    pickle.dump(vocabulary, file)

