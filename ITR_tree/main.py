from pymongo import MongoClient
import torch
from build_WordEmbedding import build_WordEmbedding
from build_tree_and_vocabulary import build_tree_and_vocabulary
from build_vocabulary import Vocabulary
from data_process import process_entry
import time
from train import train_and_eval
import pickle
from traverse_tree import traverse_tree 
import datetime
from build_ITR_tree import build_ITR_tree
from tokenize_text import tokenize_tree
import numpy as np
import psutil
import os

def main():
    
    # 连接数据库，读取数据，
    client = MongoClient('mongodb://b515:sqwUiJGHYQTikv6z@10.12.46.33:27018/?authMechanism=DEFAULT')
    dbtest = client["geth"]
    collection = dbtest.get_collection("5m_10m_2w_output")
    data = collection.find().limit(500)
    data_onehot = collection.find().limit(500)

    # 处理交易时要用的一些标量
    tree_node_list = [] # 把node放入list中
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
    
    # 开始处理每一个交易，输出一个交易嵌入，[token总数目,64]
    for idx,entry in enumerate(data):
        tx_embedding = torch.zeros(0,64)
        # 交易个数+1
        data_i +=1

        # 预处理交易，把交易变成树，生成词汇表
        Seqsstate_1, Seqslog_1, Seqscall_1 = process_entry(entry)
        root_node, vocabulary = build_tree_and_vocabulary(Seqsstate_1, Seqslog_1, Seqscall_1,vocabulary) # node.children [(结点,call),(),()] node.data ['token','token']

        #vocab_size = vocabulary.index + 1
        d_model = 64

        build_WordEmbedding(root_node, vocabulary, vocabulary.index, d_model)
        traverse_tree(root_node, tree_node_list)  # 将node节点加入到列表中

        # 遍历树的所有结点，将结点下面的矩阵全部拼接在一起
        # 11.8 增加onthot编码
        while root_node.children:
            # 先放入根结点的embeddding
            tx_embedding = torch.cat((tx_embedding,root_node.embedding),dim=0)
               
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
        
        # 将交易嵌入保存在tensor_list
        tensor_list[f"tensor_{tensor_i}"] = tx_embedding
        tensor_i += 1

        # 每隔几个交易将embedding保存到文件
        if data_i % 32 == 0 :

            file_path = f"../tmp/embedding/tree_node_list{file_i}.pth"
            with open(file_path, "w") as file:
                torch.save(tensor_list,file_path)

            tree_node_list = []
            tensor_list = {}

            file_i += 1

            tensor_i = 0

        # file_name = "./data/token_count.txt"
        # with open(file_name, "a") as file:
        #     file.write(str(tx_token_count) + "\n")
        # # 计算最终最大的交易token数量
        # if tx_token_max<tx_token_count:
        #      tx_token_max = tx_token_count

        # 每个交易结束将交易token数量清0
        # tx_token_count = 0
        #测试内存代码
        #process = psutil.Process(os.getpid())
        #memory_info = process.memory_info()
        # 输出以兆字节为单位的内存占用情况
        #print(f"Memory Usage: {memory_info.rss / (1024 * 1024):.2f} MB")
    for idx,entry in enumerate(data_onehot):
        one_hot = []
        data_i_onehot +=1

        Seqsstate_1, Seqslog_1, Seqscall_1 = process_entry(entry)
        root_node = build_ITR_tree(Seqsstate_1, Seqslog_1, Seqscall_1)
        tokenize_tree(root_node)
        
        while root_node.children:
            for token in root_node.data:
                #zero_list = [0] * vocabulary.index
                #zero_list[vocabulary.word_to_index[token]] = 1
                one_hot.append(vocabulary.word_to_index[token])
               
            # 再判断根结点下面有没有call
            if root_node.children[0][0].tag == 'call':
                for i in  range(len(root_node.children)):
                    if i ==0:
                        continue
                    else:
                        
                        for token in root_node.children[i][0].data:
                            #zero_list = [0] * vocabulary.index
                            #zero_list[vocabulary.word_to_index[token]] = 1
                            one_hot.append(vocabulary.word_to_index[token])
                root_node = root_node.children[0][0]
            else:
                for i in range(len(root_node.children)):
                    for token in root_node.children[i][0].data:
                            
                            #zero_list = [0] * vocabulary.index
                            #zero_list[vocabulary.word_to_index[token]] = 1
                            one_hot.append(vocabulary.word_to_index[token])
                break
        
        # 将one_hot如果小于256个，就增加至256个，如果大于256个，就截断
        while len(one_hot)<256:
            # zero_list = [0] * vocabulary.index
            # zero_list[0] = 1
            one_hot.append(0)
        if len(one_hot)>256:
            one_hot = one_hot[:256]
        #one_hot_arr  = np.eye(vocabulary.index)[one_hot]
        #one_hot = torch.tensor(one_hot)
        #one_hot_arr = torch.from_numpy(one_hot_arr)
        one_hot_list[f"tensor_{tensor_i_onehot}"] = one_hot
        #one_hot = []

        tensor_i_onehot += 1

        if data_i_onehot % 32 == 0 :

            file_path = f"../tmp/one_hot/one_hot_list{file_i_onehot}.pth"
            with open(file_path, "w") as file:
                torch.save(one_hot_list,file_path)

            file_i_onehot += 1
            tensor_i_onehot = 0
            one_hot_list ={}

    # 将词汇表保存到文件
    vocabulary_file = "../vocabulary/test.pkl"
    with open(vocabulary_file, "wb") as file:
        pickle.dump(vocabulary, file)

    # 记录embeeding时间，同时记下输出词汇表的地址
    end_time = time.time()
    execution_time = end_time - start_time

    file_name = "../log/log0.txt"
    with open(file_name, "a") as file:
        file.write('embedding time:    '+ str(execution_time) + "\n")
        file.write('vocabulary_file:    '+ vocabulary_file + "\n")
        file.write('vocabulary_count:    '+ str(vocabulary.index) + "\n")
  
    """
    file_i = 732
    data_i = 23392
    vocabulary_file = "./vocabulary/2w_new.pkl"
    """
    
    
    train_and_eval(file_i,data_i,vocabulary_file)
   

if __name__ == '__main__':
    main()