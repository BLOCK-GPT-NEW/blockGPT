from torch.utils.data import Dataset, DataLoader
import torch

# 自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data_files,vocabulary):
       
        self.data_files = data_files
        self.vocabulary_size = vocabulary.index
       
    def __len__(self):

        return len(self.data_files)*32

    def __getitem__(self, idx):
        # 根据dataloader要求加载的数据去文件找对应的embedding和onehot，同时对embedding大小做个补齐
        file_i = int(idx/32)
        file_i_idx = idx %32

        load_data = torch.load(f'../tmp/embedding/tree_node_list{file_i}.pth')
        tensor = load_data[f'tensor_{file_i_idx}']
        if tensor.shape[0]<256:
            padding = 256 - tensor.shape[0]
            padding_tensor = torch.zeros(padding, 64)
            tensor = torch.cat((tensor, padding_tensor), dim=0)
        elif tensor.shape[0]>256:
            tensor = tensor[:256, :]

        one_hot = torch.load(f'../tmp/one_hot/one_hot_list{file_i}.pth')
        one_hot_list = one_hot[f'tensor_{file_i_idx}']
        one_hot_arr = torch.zeros([256,self.vocabulary_size],dtype = torch.float32)
        for i in range(len(one_hot_list)):
            one_hot_arr[i][one_hot_list[i]]= 1

        
        return tensor,one_hot_arr
       