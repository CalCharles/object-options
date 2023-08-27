import numpy as np
import torch

batch = 2
keys = 3
queries = 4
heads = 5
model = 2

v = torch.tensor(np.arange(batch*keys*queries*heads*model))
v = v.reshape(batch,keys,queries,heads*model)
v[0,0] = 0
v[1,:,0] = 0
print("initial", v)
v = v.reshape(batch, keys * queries, -1)
print("conv reshape", v)
v = v.reshape(batch, keys * queries, heads, model).transpose(1,2)
print("head reshape", v)
v = v.transpose(-1,-2).reshape(batch, heads, model, keys, queries).transpose(2,3).transpose(3,4) # batch x num_heads x num_keys x num_queries x model_dim
print("key_query reshape", v)