# as of 2017-08-14, windows version only available via conda install -c peterjc123 pytorch

# CUDA TEST
import torch
from torch.autograd import Variable
from torch import nn

x = torch.Tensor([1.0])
xx = x.cuda()
print(xx)

# CUDNN TEST
from torch.backends import cudnn
print(cudnn.is_acceptable(xx))

word_embedding = nn.Embedding(10, 300).cuda()
bio_embedding = nn.Embedding(10, 32).cuda()

# a batch of 2 samples of 4 indices each
word_input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]).cuda())
bio_input = Variable(torch.LongTensor([[1,2,4,5],[4,3,2,9]]).cuda())
wb = word_embedding(word_input)
bb = bio_embedding(bio_input)

input_emd = torch.cat((wb, bb), dim=2)
print(input_emd.size())
loss = input_emd.sum()
print(loss)
loss.backward()