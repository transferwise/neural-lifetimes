# # dont think this module is used at all
# from typing import List

# from torch import nn as nn


# class MLP(nn.Module):
#     def __init__(self, layers: List[int], drop_rate=0.0):
#         super().__init__()
#         self.layers = nn.ModuleList()
#         for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
#             self.layers.append(nn.Linear(in_dim, out_dim))
#             self.layers.append(nn.Dropout(drop_rate))
#             self.layers.append(nn.ReLU())
#         self.layers.append(nn.Linear(layers[-2], layers[-1]))

#     def forward(self, x):
#         batch_size = x["target"].shape[0]
#         x_in = x["data"].reshape(batch_size, -1)
#         for i, layer in enumerate(self.layers):
#             x_in = layer(x_in)

#         x["output"] = x_in
#         return x

# TODO delete this file or uncomment it
