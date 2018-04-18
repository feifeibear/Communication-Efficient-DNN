import torch
import numpy as np
from torch.autograd import Variable

#x = torch.FloatTensor([[1, 2, 3], [4, 5, 6]])

def prune_perc(x, perc):
    x_size = x.size()
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    x_flatten = torch.abs(x.view(x_len))
    # torch.save(x_flatten, './tensors/x_flatten.pt')
    top_k = int(x_len * perc)
    #print(x)
    #print("x_flatten norm : ", x_flatten.norm())
    #print("x_len : ", x_len, "debug top_k : ", top_k)

    if top_k < 1:
        top_k = 1
    _, x_top_idx = torch.topk(x_flatten, top_k, 0, largest=True)
    #x_top_idx,_ = torch.sort(x_top_idx)
    #print('nonzero indices are : ', x_top_idx)
    # torch.save(x_top_idx, './tensors/x_top_idx.pt')

    # for i in x_top_idx:
    #     if i >= x_len:
    #         print("Error in top_k", "idx : ", i, " len : ", x_len)
    # x_top_idx = torch.LongTensor([i for i in range(x_len)]).cuda()
    # print(x_top_val, x_top_idx)

    if torch.cuda.is_available():
        mask = torch.zeros(x_len).cuda()
    else:
        mask = torch.zeros(x_len)

    mask[x_top_idx] = 1.0
    mask = mask.view(x_size)
    return mask


def check_sparsity(x):
    nnz = len(torch.nonzero(x == 0.))
    x_len = 1;
    for dim in x.size():
        x_len *= dim
    return 1- nnz / x_len
    #print(mask)
#grad = torch.sparse.FloatTensor(x_top_idx, x_top_val, torch.Size([x_len]))#.to_dense().view(x_size)
#print(grad)
#residue = x - grad
#print(grad, residue)

if __name__ == '__main__':
    torch.manual_seed(123)
    x = torch.randn(5,5) #FloatTensor([[1, 2, 3], [4, 5, 6]])
    #x = torch.randn(5) #FloatTensor([[1, 2, 3], [4, 5, 6]])
    x = x.cuda()
    mask = prune_perc(x, 0.2)
    print(x)
    print(x * mask)
    print(x * (1. - mask))

    sx = (x * (1. - mask))
    print("sparse ", check_sparsity(x))
    print("sparse ", check_sparsity(x * mask))
