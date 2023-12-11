import torch
test =torch.load('./tmp/test_embedding/tree_node_list0.pth')
print(test['tensor_0'].shape)