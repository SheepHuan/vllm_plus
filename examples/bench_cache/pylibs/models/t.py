import torch


a = torch.load("past_key_values.pth")
b = torch.load("old_kvs.pth")


for i in range(len(b)):
    for j in range(len(b[i])):
        len = b[i][j].shape[2]
        gt = a[i][j][:,:,:len,:]
        pred = b[i][j][:,:,:len,:]
        pass

pass