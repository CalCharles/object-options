from Network.network import Network
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
from Network.network_utils import reduce_function, get_acti
from Network.General.mlp import MLPNetwork
from Network.General.conv import ConvNetwork

class PairNetwork(Network):
    def __init__(self, args):
        super().__init__(args)
        # assumes the input is flattened list of input space sized values
        # needs an object dim
        # does NOT require a fixed number of objects, because it collects through a max operator
        self.object_dim = args.pair.object_dim
        self.first_obj_dim = args.pair.first_obj_dim
        self.post_dim = args.pair.post_dim
        self.drop_first = args.drop_first if 'drop_first' in args else False
        self.reduce_fn = args.pair.reduce_function
        self.difference_first = args.pair.difference_first
        self.num_layers = args.pair.num_pair_layers
        self.conv_object_dim = args.pair.object_dim + max(0, self.first_obj_dim * int(not self.drop_first)) + args.pair.object_dim * int(args.pair.difference_first)
        self.conv_dim = self.hs[-1] + max(0, self.post_dim) if args.pair.aggregate_final and len(self.hs) > 0 else args.num_outputs
        no_nets = args.pair.no_nets if 'no_nets' in args.pair else False
        

        layers = list()
        self.layer_conv_dim = self.hs[-1] if args.pair.aggregate_final and len(self.hs) > 0 else args.num_outputs
        if not no_nets:
            if len(self.hs) != 0:
                encode_layer_args = copy.deepcopy(args)
                encode_layer_args.object_dim = self.conv_object_dim
                encode_layer_args.include_last = False
                encode_layer_args.output_dim = self.layer_conv_dim
                encode_layer_args.hidden_sizes = list()
                self.encode_layer = ConvNetwork(encode_layer_args)
                layers += [self.encode_layer]

                self.conv_layers = list()
                conv_args = copy.deepcopy(args)
                conv_args.object_dim = self.layer_conv_dim
                conv_args.output_dim = self.layer_conv_dim
                conv_args.activation_final = conv_args.activation
                conv_args.include_last = True
                self.conv_args = conv_args
                # print (self.layer_conv_dim, self.hs[-1], args.num_outputs, self.conv_object_dim)
                for i in range(self.num_layers):
                    self.conv_layers.append(ConvNetwork(conv_args))
                self.conv_layers = nn.ModuleList(self.conv_layers)
                layers.append(self.conv_layers)

            decode_layer_args = copy.deepcopy(args)
            if len(self.hs) == 0: decode_layer_args.object_dim = self.conv_object_dim
            else: decode_layer_args.object_dim = self.layer_conv_dim
            decode_layer_args.include_last = not args.pair.aggregate_final
            decode_layer_args.activation_final = decode_layer_args.activation if args.pair.aggregate_final else decode_layer_args.activation_final
            decode_layer_args.output_dim = self.conv_dim
            decode_layer_args.hidden_sizes = list()
            self.decode_layer = ConvNetwork(decode_layer_args)
            layers += [self.decode_layer]

        post_mlp_args = copy.deepcopy(args) 
        if args.pair.post_dim > 0:
            args.num_inputs = args.post_dim + args.first_obj_dim
            args.num_outputs = self.hs[-1] if args.pair.aggregate_final and len(self.hs) > 0 else args.num_outputs
            if not no_nets:
                self.post_channel = MLPNetwork(args)
                layers.append(self.post_channel)
        self.aggregate_final = args.pair.aggregate_final
        # self.softmax = nn.Softmax(-1)
        if args.pair.aggregate_final: # does not work with a post-channel
            args.include_last = True
            args.num_inputs = self.conv_dim
            args.num_outputs = self.num_outputs
            args.hidden_sizes = args.pair.final_layers # TODO: hardcoded final hidden sizes for now
            if not no_nets: 
                self.MLP = MLPNetwork(args)
                layers.append(self.MLP)
        self.model = layers
        self.train()
        self.reset_network_parameters()
        # print(self)
        # error
        # print(self, self.first_obj_dim, self.object_dim)

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
        # print(self.object_dim, x.shape[-1])
        nobj = x.shape[-1] // self.object_dim
        x = x.view(-1, nobj, self.object_dim)
        if self.first_obj_dim > 0 and not self.drop_first:
            # append the pre components to every object and reshape
            broadcast_fx = torch.stack([fx.clone() for i in range(nobj)], dim=len(fx.shape) - 1)
             # appends the difference between the last of pre-comp and object, fx must be at least as long as x, 
            if self.difference_first:
                dx = x - torch.stack([fx[...,fx.shape[-1]-x.shape[-1]:].clone() for i in range(nobj)], dim=len(fx.shape) - 1)
                broadcast_fx = torch.cat((broadcast_fx, dx), dim = -1) 
            # print(broadcast_fx[0], x[0])
            x = torch.cat((broadcast_fx, x), dim=-1)
        # transpose because conv-nets have reversed dimensions
        x = x.transpose(-1,-2)
        return x, fx, px, batch_size

    def run_networks(self, x, px, batch_size):
        if len(self.hs) > 0:
            # print(x[0], x.shape)
            x = self.encode_layer(x)
            # print("pair", x[0], x.shape, self.conv_layers, self.layer_conv_dim)
            for c_layer in self.conv_layers:
                x = c_layer(x)
                # print("layer", x[0])
        # print("pair", x.shape, self.aggregate_final, self.num_outputs, self.decode_layer)
        x = self.decode_layer(x)
        # print("pair", x.shape, self.aggregate_final, self.num_outputs)
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
        # print("pair", x.shape)
        return x

    def forward(self, x):
        x, fx, px, batch_size = self.slice_input(x)
        x = self.run_networks(x, px, batch_size)
        return x
