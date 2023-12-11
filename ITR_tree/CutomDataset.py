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

        file_i = int(idx/32)
        file_i_idx = idx %32

        load_data = torch.load(f'../tmp/test_new_1/embedding/tree_node_list{file_i}.pth')
        tensor = load_data[f'tensor_{file_i_idx}']

        one_hot = torch.load(f'../tmp/test_new_1/one_hot/one_hot_list{file_i}.pth')
        one_hot_list = one_hot[f'tensor_{file_i_idx}']

        src_key_mask = one_hot_list == 0
        
        return tensor,one_hot_list,src_key_mask
       
    
class CustomDataset_det(Dataset):
    def __init__(self, data_files,vocabulary):
       
        self.data_files = data_files
        self.vocabulary_size = vocabulary.index

    def __len__(self):

        return len(self.data_files)*32

    def __getitem__(self, idx):

        file_i = int(idx/32)
        file_i_idx = idx %32

        load_data = torch.load(f'../tmp/test_new_1/test_embedding/tree_node_list{file_i}.pth')
        tensor = load_data[f'tensor_{file_i_idx}']

        one_hot = torch.load(f'../tmp/test_new_1/test_one_hot/one_hot_list{file_i}.pth')
        one_hot_list = one_hot[f'tensor_{file_i_idx}']

        src_key_mask = one_hot_list == 0
        
        return tensor,one_hot_list,src_key_mask
       