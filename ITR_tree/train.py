import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from CutomDataset import CustomDataset,CustomDataset_det
from transformer_encoder import generate_square_subsequent_mask
from transformer_encoder import TransformerEncoderModel
import torch
from torch import nn, Tensor
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time
import pickle
import torch.nn.functional as F

def train_and_eval(file_i,data_i,vocabulary_file):    
    # 从文件中加载词汇表
    with open(vocabulary_file,"rb") as file:   
        vocabulary = pickle.load(file)
    
    # 计算有多少个文件，一个文件里面32个交易
    embedding_file = []
    for i in range (file_i):
        embedding_file.append(f"../tmp/test_new_1/embedding/tree_node_list{i}.pth")

    custom_data = CustomDataset(embedding_file,vocabulary)
    custom_train, custom_val = train_test_split(custom_data, test_size=0.2, random_state=42)

    #创建数据加载
    batch_size = 32
    num_workers = 0
    train_data_loader = DataLoader(custom_train, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=num_workers)
    val_data_loader = DataLoader(custom_val, batch_size=batch_size, shuffle=True,pin_memory=True,num_workers=num_workers)

    # ========================训练部分==========================
    # 使用预处理好的 embedding 初始化 nn.Embedding
    # 注意：需要根据你的预训练 embedding 的维度来设置 embed_size
    # embedding_layer = train_data_loader  # 替换成你的预训练 embedding 我们的embedding直接在处理交易的时候就弄好了
    
    # 初始化 Transformer 编码器模型
    hidden_size = 256
    num_heads = 8
    num_layers = 6
    d_model = 64
    
    # 将模型送入gpu
    device = torch.device('cuda')
    encoder_model = TransformerEncoderModel(vocabulary.index,d_model, hidden_size, num_heads, num_layers).to(device)
    #encoder_model = torch.load('../model/random.pth')
    encoder_model.to(device)

    # 保存模型参数
    state_dict = encoder_model.state_dict()
    param = torch.cat([(state_dict[name].flatten())for name in state_dict]).to('cuda')
    with open("../log/log0.txt", "a") as file:
        file.write('param:    '+ str(param.shape[0]) + "\n")


    # 定义损失函数和优化器
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(encoder_model.parameters(), lr=0.001)

    # 创建一个学习率调度器，比如StepLR
    #scheduler = lr_scheduler.StepLR(optimizer, step_size=2, gamma=0.5)
    epochs = 3

    # 保存相关信息到日志中
    file_name = "../log/log0.txt"
    with open(file_name, "a") as file:
        file.write('batch_size:    '+ str(batch_size) + "\n")
        file.write('eopchs:    '+ str(epochs) + "\n")
        file.write('data amounts:    '+ str(data_i) + "\n")

    # 顺序掩码，用于只看当前及以前的内容
    src_mask = generate_square_subsequent_mask(64).to(device)
    
    # 开始每一轮
    for epoch in range(epochs):
        # 记录训练开始时间
        start_time = time.time()

        print('开始训练')
        encoder_model.train()
        train_loss=0
        batch_i = 0
        # 无监督训练
        for batch ,one_hot,src_key_mask  in train_data_loader:
            optimizer.zero_grad()
            batch, one_hot ,src_key_mask= batch.to(device), one_hot.to(device),src_key_mask.to(device)
            outputs = encoder_model(batch,src_mask,src_key_mask)
            loss = loss_function(outputs.view(-1,vocabulary.index), one_hot.reshape(-1))
            train_loss+=loss.item()
            loss.backward()
            if batch_i%20 ==0:
                print(f'第{batch_i}轮训练')
                print(loss.item())
            optimizer.step()
            batch_i += 1
        
        print('开始验证')
        encoder_model.eval()
        val_loss = 0
        with torch.no_grad():
            i= 0
            batch_i=0
            for batch ,one_hot,src_key_mask  in val_data_loader:
                    batch, one_hot ,src_key_mask= batch.to(device), one_hot.to(device),src_key_mask.to(device)
                    outputs = encoder_model(batch,src_mask,src_key_mask)
                    loss = loss_function(outputs.view(-1,vocabulary.index), one_hot.reshape(-1))
                    val_loss += loss.item()
                    if batch_i%20 ==0:
                        print(f'第{batch_i}轮验证')
                        print(loss.item())
                    batch_i+=1
        # 更新动态学习率            
        #scheduler.step()

        # 打印每一轮信息
        print(f"Epoch {epoch+1}/{epochs} - Validation Loss: {val_loss/len(val_data_loader)}" + f"   train Loss: {train_loss/len(train_data_loader)}")
        print(f"第 {epoch+1}/{epochs} 轮完成")

        # 保存相关信息
        end_time = time.time()
        execution_time = end_time - start_time
        file_name = "../log/log0.txt"
        with open(file_name, "a") as file:
                file.write(f'epoch:{epoch+1}'+"    "+"time spend:"+str(execution_time )+'    '+'train Loss:'+f'{train_loss/len(train_data_loader)}'+'    '+'Validation Loss:'+ f'{val_loss/len(val_data_loader)}'+'\n')
      
        # 保存模型
        model_file = '../model/random_new_1_1.pth'
        torch.save(encoder_model, model_file)

    # 训练完成后，保存相关信息
    file_name = "../log/log0.txt"
    with open(file_name, "a") as file:
        current_datetime = datetime.datetime.now()
        formatted_time = current_datetime.strftime("%Y-%m-%d %H:%M:%S")

        file.write('end date:    '+ formatted_time + "\n")
        file.write('model file:    '+ model_file + "\n")
        file.write('--------------------------------------------'+"\n")

    print('结束')