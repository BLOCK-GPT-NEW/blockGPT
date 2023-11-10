import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from CutomDataset import CustomDataset
from transformer_encoder import generate_square_subsequent_mask
from transformer_encoder import TransformerEncoderModel
import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import pickle
from pymongo import MongoClient

# def generate_square_subsequent_mask(sz):
#     """生成一个负无穷大的上三角矩阵，对角线上元素为 0。"""
#     return torch.triu(torch.ones(sz, sz) * float('-inf'), diagonal=1)

def train_and_eval(file_i,data_i,vocabulary_file):    
    # 从文件中加载词汇表
    with open(vocabulary_file,"rb") as file:   
        vocabulary = pickle.load(file)
   
    embedding_file_train = []
    
    for i in range (file_i):
        embedding_file_train.append(f"../tmp/embedding/tree_node_list{i}.pth")

    # 划分数据集、验证集
    custom_train = CustomDataset(embedding_file_train,vocabulary)
    # train_sequences, val_sequences = train_test_split(custom_data, test_size=0.1, random_state=42)

    #创建数据加载器
    batch_size = 32
    num_workers = 0
    train_data_loader = DataLoader(custom_train, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=num_workers)
    # val_data_loader = DataLoader(custom_val, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=num_workers)

    # ========================训练部分==========================
    # 使用预训练好的 embedding 初始化 nn.Embedding
    # 注意：需要根据你的预训练 embedding 的维度来设置 embed_size
    # embedding_layer = train_data_loader  # 替换成你的预训练 embedding 我们的embedding直接在处理交易的时候就弄好了
    
    # 初始化 Transformer 编码器模型
    hidden_size = 2056
    num_heads = 8
    num_layers = 6
    d_model = 64

    device = torch.device('cuda')
    encoder_model = TransformerEncoderModel(vocabulary.index,d_model, hidden_size, num_heads, num_layers).to(device)
    encoder_model.to(device)

    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(encoder_model.parameters(), lr=0.0001)
    # 创建一个学习率调度器，比如StepLR
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.5)
    epochs = 5

    # 保存相关信息到日志中
    file_name = "../log/log0.txt"
    with open(file_name, "a") as file:
        file.write('batch_size:    '+ str(batch_size) + "\n")
        file.write('eopchs:    '+ str(epochs) + "\n")
        file.write('data amounts:    '+ str(data_i) + "\n")

    # 开始每一轮
    for epoch in range(epochs):
        # 记录训练开始时间
        start_time = time.time()

        print('开始训练')
        encoder_model.train()
        train_loss=0
        batch_i = 0
     
        src_mask = generate_square_subsequent_mask(256).to(device)
       
        # 无监督与训练，不需要标签，不需要验证集
        for batch ,one_hot in train_data_loader:
         
            optimizer.zero_grad()
            batch, one_hot = batch.to(device), one_hot.to(device)
            batch = batch.transpose(0,1)
            outputs = encoder_model(batch,src_mask)

            outputs = outputs.transpose(0,1)
            loss = loss_function(outputs, one_hot)
            train_loss+=loss.item()
            loss.backward()

            print(f'第{batch_i}轮训练')
            print(loss.item())
            optimizer.step()

            batch_i += 1

        # for inputs, targets in train_data_loader:

        #     optimizer.zero_grad()
        #     inputs, targets = inputs.to(device), targets.to(device)
 
        #     outputs = encoder_model(inputs)

        #     targets = torch.argmax(targets, dim=-1)
    
        #     loss = loss_function(outputs.permute(0, 2, 1), targets)
        #     print(f'第{batch_i}轮训练')
        #     print(loss.item())
        #     train_loss+=loss.item()
        #     loss.backward(retain_graph=True)
            
        #     optimizer.step()
        #     scheduler.step()

        #     batch_i += 1
        """
        print('开始验证')
        # Validate此处要替换为验证集合
        encoder_model.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs, targets in val_data_loader:

                inputs, targets = inputs.to(device), targets.to(device)
                outputs = encoder_model(inputs)
                targets = torch.argmax(targets, dim=-1)
                loss = loss_function(outputs.permute(0, 2, 1), targets)
                val_loss += loss.item()
         """

        
        #print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss/len(val_data_loader)}" + f"   train Loss: {train_loss/len(train_data_loader)}")
        print(f"Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_data_loader)}")
        print(f"第 {epoch+1}/{epochs} 轮完成")

        # 保存相关信息
        end_time = time.time()
        execution_time = end_time - start_time
        file_name = "../log/log0.txt"
        with open(file_name, "a") as file:
                #file.write(f'epoch:{epoch+1}'+"    "+"time spend:"+str(execution_time )+'    '+'train Loss:'+f'{train_loss/len(train_data_loader)}'+'Validation Loss:'+ f'{val_loss/len(val_data_loader)}'+'\n')
                file.write(f'epoch:{epoch+1}'+"    "+"time spend:"+str(execution_time )+'    '+'train Loss:'+f'{train_loss/len(train_data_loader)}'+'\n')
    
    # 保存模型
    model_file = '../model/2w_new.pth'
    torch.save(encoder_model, model_file)

    # 保存相关信息
    file_name = "../log/log0.txt"
    with open(file_name, "a") as file:
        current_datetime = datetime.datetime.now()
        formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        file.write('end date:    '+ formatted_time + "\n")
        file.write('model file:    '+ model_file + "\n")
        file.write('--------------------------------------------'+"\n")

    print('结束')



