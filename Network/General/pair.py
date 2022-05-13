from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from Network.network_utils import reduce_function
from Network.General.mlp import MLPNetwork
from Network.General.conv import ConvNetwork

class PairNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        # assumes the input is flattened list of input space sized values
        # needs an object dim
        # does NOT require a fixed number of objects, because it collects through a max operator
        self.object_dim = args.object_dim
        self.first_obj_dim = args.first_obj_dim
        self.post_dim = args.post_dim
        self.drop_first = args.drop_first if 'drop_first' in args else False
        self.reduce_fn = args.reduce_function
        
        conv_args = copy.deepcopy(args)
        conv_args.object_dim += max(0, self.first_obj_dim * int(not self.drop_first))
        self.conv_dim = self.hs[-1] if len(self.hs) > 0 else args.output_dim    
        if args.aggregate_final: conv_args.output_dim = self.hs[-1] + max(0, self.post_dim) 
        conv_args.include_last = args.aggregate_final

        layers = list()
        self.conv = ConvNetwork(conv_args)
        layers.append(self.conv)

        post_mlp_args = copy.deepcopy(kargs) 
        if args.post_dim > 0:
            args.num_inputs = args.post_dim + args.first_obj_dim
            args.num_outputs = self.hs[-1]
            self.post_channel = MLPNetwork(args)
            layers.append(self.post_channel)
        self.aggregate_final = args.aggregate_final
        self.activation_final = args.activation_final if "activation_final" in args else ""
        self.softmax = nn.Softmax(-1)
        if args.aggregate_final: # does not work with a post-channel
            args.include_last = True
            args.num_inputs = conv_args.output_dim
            args.num_outputs = self.num_outputs
            args.hidden_sizes = [256] # TODO: hardcoded final hidden sizes for now
            self.MLP = MLPNetwork(args)
            layers.append(self.MLP)
        self.model = layers
        self.train()
        self.reset_parameters()

    def slice_input(self, x):
        fx, px = None, None
        # input of shape: [batch size, ..., flattened state shape]
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        output_shape = x.shape[-1] - self.first_obj_dim  - self.post_dim
        if self.post_dim > 0:
            # cut out the "post" component, which is sent through a different channel
            px = torch.cat([x[...,:self.first_obj_dim], x[..., x.shape[-1]-self.post_dim:]], dim=-1)
            px = px.view(-1, px.shape[-1])
        if self.first_obj_dim > 0:
            # cut out the "pre" component, which is appended to every object
            fx = x[..., :self.first_obj_dim] # TODO: always assumes first object dim is the first dimensions
            fx = fx.view(-1, self.first_obj_dim)
            # cut out the object components
            x = x[..., self.first_obj_dim:x.shape[-1]-self.post_dim]

        # reshape the object components
        nobj = x.shape[-1] // self.object_dim
        x = x.view(-1, nobj, self.object_dim)
        if self.first_obj_dim > 0 and not self.drop_first:
            # append the pre components to every object and reshape
            broadcast_fx = torch.stack([fx.clone() for i in range(nobj)], dim=len(fx.shape) - 1)
            x = torch.cat((broadcast_fx, x), dim=-1)
        # transpose because conv-nets have reversed dimensions
        x = x.transpose(-1,-2)
        return x, fx, px, batch_size

    def run_networks(self, x, px, batch_size):
        x = self.conv(x)
        if self.aggregate_final:
            # combine the conv outputs using the reduce function, and append any post channels
            x = reduce_function(self.reduce_fn, x)
            x = x.view(-1, self.conv_dim)
            if self.post_dim > 0:
                px = self.post_channel(px)
                x = torch.cat([x,px], dim=-1)

            # final network
            x = self.MLP(x)
        else:
            # when dealing without many-many input outputs
            x = x.transpose(2,1)
            x = x.reshape(batch_size, -1)
        x = self.activation_final(x)
        return x

    def forward(self, x):
        x, fx, px, batch_size = self.slice_input(x)
        x = self.run_networks(x, px, batch_size)
        return x
