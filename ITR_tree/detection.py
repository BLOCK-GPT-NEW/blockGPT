import torch
from train import generate_square_subsequent_mask
import pickle
from torch.utils.data import Dataset, DataLoader
from CutomDataset import CustomDataset
import torch.nn as nn
import csv
from pymongo import MongoClient
import pymongo

# 连接数据库，读取数据，
client = MongoClient('mongodb://b515:sqwUiJGHYQTikv6z@10.12.46.33:27018/?authMechanism=DEFAULT')
dbtest = client["geth"]
collection = dbtest.get_collection("5m_10m_2w_output")

file_i = 732
data_i = 23424
vocabulary_file = "../vocabulary/2w_new.pkl"

with open(vocabulary_file,"rb") as file:   
        vocabulary = pickle.load(file)
encoder_model = torch.load('../model/2w_new.pth')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
embedding_file_train = []
for i in range (file_i):
    embedding_file_train.append(f"../tmp/embedding/tree_node_list{i}.pth")

custom_train = CustomDataset(embedding_file_train,vocabulary)
train_data_loader = DataLoader(custom_train, batch_size=1, shuffle=False,pin_memory=True)

encoder_model.eval()  # 开启评估模式
loss_function = nn.CrossEntropyLoss()
src_mask = generate_square_subsequent_mask(256).to(device)

# 指定要写入的CSV文件路径
csv_file_path = '../detection_result/2w_new.csv'
# 打开CSV文件并写入数据
with open(csv_file_path, 'w', newline='') as csv_file:
   
    with torch.no_grad():
        i= 0
        for batch ,one_hot in train_data_loader:
                batch, one_hot = batch.to(device), one_hot.to(device)
                batch = batch.squeeze()
                one_hot = one_hot.squeeze()

                outputs = encoder_model(batch,src_mask)

                loss = loss_function(outputs, one_hot)

                result = collection.find_one(skip=i, sort=[('_id', pymongo.ASCENDING)])
                i +=1
                csv_writer = csv.writer(csv_file)

                # 使用写入器对象写入数据
                csv_writer.writerow([result['tx_hash'],loss.item()])
