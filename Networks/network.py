import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

class pytorch_model():
    def __init__(self, combiner=None, loss=None, reducer=None, cuda=True):
        # should have customizable combiner and loss, but I dont.
        self.cuda=cuda
        self.reduce_size = 2 # someday this won't be hard coded either

    @staticmethod
    def wrap(data, dtype=torch.float, cuda=True, device = None):
        # print(Variable(torch.Tensor(data).cuda()))
        if type(data) == torch.Tensor:
            if cuda: # TODO: dtype not handeled 
                return data.clone().detach().cuda(device=device)
            else:
                return data.clone().detach()
        else:
            if cuda:
                return torch.tensor(data, dtype=dtype).cuda(device=device)
            else:
                return torch.tensor(data, dtype=dtype)

    @staticmethod
    def unwrap(data):
        if type(data) == torch.Tensor:
            return data.clone().detach().cpu().numpy()
        else:
            return data

    @staticmethod
    def concat(data, axis=0):
        return torch.cat(data, dim=axis)

# # normalization functions
class NormalizationFunctions():
    def __init__(self, **kwargs):
        pass

    def __call__(self, val):
        return

    def reverse(self, val):
        return

class ConstantNorm(NormalizationFunctions):
    def __init__(self, **kwargs):
        self.mean = kwargs['mean']
        self.std = kwargs['variance']
        self.inv_std = kwargs['invvariance']

    def __call__(self, val):
        # print(val, self.mean, self.inv_std)
        return (val - self.mean) * self.inv_std

    def reverse(self, val):
        return val * self.std + self.mean

    def cuda(self):
        if type(self.mean) == torch.Tensor:
            self.mean = self.mean.cuda()
            self.std = self.std.cuda()
            self.inv_std = self.inv_std.cuda()

    def cpu(self):
        if type(self.mean) == torch.Tensor:
            self.mean = self.mean.cpu()
            self.std = self.std.cpu()
            self.inv_std = self.inv_std.cpu()

## end of normalization functions

class Network(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.num_inputs, self.num_outputs = kwargs["num_inputs"], kwargs["num_outputs"]
        self.init_form = kwargs["init_form"]
        self.layers = []
        self.acti = self.get_acti(kwargs["activation"])
        self.iscuda = False

    def cuda(self):
        super().cuda()
        self.iscuda = True

    def cpu(self):
        super().cpu()
        self.iscuda = False


    def run_acti(self, acti, x):
        if acti is not None:
            return acti(x)
        return x

    def get_acti(self, acti):
        if acti == "relu":
            return F.relu
        elif acti == "sin":
            return torch.sin
        elif acti == "sigmoid":
            return torch.sigmoid
        elif acti == "tanh":
            return torch.tanh
        elif acti == "none":
            return None

    def reset_parameters(self):
        relu_gain = nn.init.calculate_gain('relu')
        for layer in self.model:
            if type(layer) == nn.Conv2d:
                if self.init_form == "orth":
                    nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                else:
                    nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu') 
            elif issubclass(type(layer), Network):
                layer.reset_parameters()
            elif type(layer) == nn.Parameter:
                nn.init.uniform_(layer.data, 0.0, 0.2/np.prod(layer.data.shape))#.01 / layer.data.shape[0])
            elif type(layer) == nn.Linear or type(layer) == nn.Conv1d:
                fulllayer = layer
                if type(layer) != nn.ModuleList:
                    fulllayer = [layer]
                for layer in fulllayer:
                    print("init form", self.init_form)
                    if self.init_form == "orth":
                        nn.init.orthogonal_(layer.weight.data, gain=nn.init.calculate_gain('relu'))
                    elif self.init_form == "uni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                         nn.init.uniform_(layer.weight.data, 0.0, 1 / layer.weight.data.shape[0])
                    elif self.init_form == "smalluni":
                        # print("div", layer.weight.data.shape[0], layer.weight.data.shape)
                        nn.init.uniform_(layer.weight.data, -.0001 / layer.weight.data.shape[0], .0001 / layer.weight.data.shape[0])
                    elif self.init_form == "xnorm":
                        torch.nn.init.xavier_normal_(layer.weight.data)
                    elif self.init_form == "xuni":
                        torch.nn.init.xavier_uniform_(layer.weight.data)
                    elif self.init_form == "knorm":
                        torch.nn.init.kaiming_normal_(layer.weight.data)
                    elif self.init_form == "kuni":
                        torch.nn.init.kaiming_uniform_(layer.weight.data)
                    elif self.init_form == "eye":
                        torch.nn.init.eye_(layer.weight.data)
                    if hasattr(layer, 'bias') and layer.bias is not None:
                        nn.init.uniform_(layer.bias.data, 0.0, 1e-6)
                    # print("layer", self.init_form)

    def get_parameters(self):
        params = []
        for param in self.parameters():
            params.append(param.data.flatten())
        return torch.cat(params)

    def get_gradients(self):
        grads = []
        for param in self.parameters():
            grads.append(param.grad.data.flatten())
        return torch.cat(grads)

    def set_parameters(self, param_val): # sets the parameters of a model to the parameter values given as a single long vector
        if len(param_val) != self.count_parameters():
            raise ValueError('invalid number of parameters to set')
        pval_idx = 0
        for param in self.parameters():
            param_size = np.prod(param.size())
            cur_param_val = param_val[pval_idx : pval_idx+param_size]
            if type(cur_param_val) == torch.Tensor:
                param.data = cur_param_val.reshape(param.size()).float().clone()
            else:
                param.data = torch.from_numpy(cur_param_val) \
                              .reshape(param.size()).float()
            pval_idx += param_size
        if self.iscuda:
            self.cuda()

    def count_parameters(self, reuse=True):
        if reuse and self.parameter_count > 0:
            return self.parameter_count
        self.parameter_count = 0
        for param in self.parameters():
            # print(param.size(), np.prod(param.size()), self.insize, self.hidden_size)
            self.parameter_count += np.prod(param.size())
        return self.parameter_count

    def forward(self, x):
        '''
        all should have a forward function, but not all forward functions have the same signature
        '''
        return

class BasicMLPNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hs = kwargs['hidden_sizes']
        self.use_layer_norm = kwargs['use_layer_norm']

        if len(self.hs) == 0:
            if self.use_layer_norm:
                layers = [nn.LayerNorm(self.num_inputs), nn.Linear(self.num_inputs, self.num_outputs)]
            else:
                layers = [nn.Linear(self.num_inputs, self.num_outputs)]
        elif self.use_layer_norm:
            layers = ([nn.LayerNorm(self.num_inputs), nn.Linear(self.num_inputs, self.hs[0]), nn.ReLU(inplace=True),nn.LayerNorm(self.hs[0])] + 
                  sum([[nn.Linear(self.hs[i-1], self.hs[i]), nn.ReLU(inplace=True), nn.LayerNorm(self.hs[i])] for i in range(1, len(self.hs))], list()) + 
                [nn.Linear(self.hs[-1], self.num_outputs)])
        else:
            layers = ([nn.Linear(self.num_inputs, self.hs[0]), nn.ReLU(inplace=True)] + 
                  sum([[nn.Linear(self.hs[i-1], self.hs[i]), nn.ReLU(inplace=True)] for i in range(1, len(self.hs))], list()) + 
                [nn.Linear(self.hs[-1], self.num_outputs)])
        self.model = nn.Sequential(*layers)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        return x

class BasicResNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hs = kwargs['hidden_sizes']
        self.use_layer_norm = kwargs['use_layer_norm']
        self.residual_rate = 2
        last_layer = nn.Linear(self.hs[-1], self.num_outputs) if len(self.hs) % 2 == 1 else nn.Linear(self.hs[-1] + self.hs[-2], self.num_outputs)
        even_layer = [nn.Linear(self.hs[-2], self.hs[-1])] if len(self.hs) % 2 == 0 else list()
        layers = ([nn.Linear(self.num_inputs, self.hs[0])] + 
              sum([[nn.Linear(self.hs[i-2], self.hs[i-1]), nn.Linear(self.hs[i-1] + self.hs[i-2], self.hs[i])] for i in [j * 2 for j in range(1, len(self.hs) // 2 + int(len(self.hs) % 2))]], list()) + 
            even_layer + [last_layer])
        self.model = nn.ModuleList(layers)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        for i, l in enumerate(self.model):
            x = l(x)
            if i != len(self.model) - 1:
                x = torch.relu(x)
            if i % 2 == 0:
                lx = x
            elif i != len(self.model) - 1:
                x = torch.cat([lx, x], dim=-1)
        return x


class BasicConvNetwork(Network): # basic 1d conv network 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hs = kwargs['hidden_sizes']
        self.use_layer_norm = kwargs['use_layer_norm']
        self.object_dim = kwargs["object_dim"]
        self.output_dim = kwargs["output_dim"]
        include_last = kwargs['include_last']

        if len(self.hs) == 0:
            layers = [nn.Conv1d(self.object_dim, self.output_dim, 1)]
        else:
            if len(self.hs) == 1:
                layers = [nn.Conv1d(self.object_dim, self.hs[0], 1)]
            elif self.use_layer_norm:
                layers = ([nn.Conv1d(self.object_dim, self.hs[0], 1), nn.ReLU(inplace=True),nn.LayerNorm(self.hs[0])] + 
                  sum([[nn.Conv1d(self.hs[i-1], self.hs[i], 1), nn.ReLU(inplace=True), nn.LayerNorm(self.hs[i])] for i in range(1, len(self.hs) - 1)], list())
                    + [nn.Conv1d(self.hs[-2], self.hs[-1], 1), nn.ReLU(inplace=True)])
            else:
                layers = ([nn.Conv1d(self.object_dim, self.hs[0], 1), nn.ReLU(inplace=True)] + 
                      sum([[nn.Conv1d(self.hs[i-1], self.hs[i], 1), nn.ReLU(inplace=True)] for i in range(1, len(self.hs) - 1)], list())
                      + [nn.Conv1d(self.hs[-2], self.hs[-1], 1), nn.ReLU(inplace=True)])
            if include_last: # if we include last, we need a relu after second to last. If we do not include last, we assume that there is a layer afterwards so we need a relu after the second to last
                layers += [nn.Conv1d(self.hs[-1], self.output_dim, 1)]
        self.model = nn.Sequential(*layers)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        x = self.model(x)
        return x

class Basic2DConvNetwork(Network): # basic 1d conv network 
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hs = kwargs['hidden_sizes']
        self.kernel = kwargs["kernel"]
        self.use_layer_norm = kwargs['use_layer_norm']
        self.input_dims = kwargs["input_dims"]
        self.channels = self.input_dims[-1] if len(self.input_dims) > 2 else 1
        self.stride = kwargs["stride"]
        self.padding = kwargs["padding"]
        self.output_dim = kwargs["output_dim"] # used for include last AND reduce, meaning there could be overlap isues
        self.reduce = kwargs["reduce"]
        include_last = kwargs['include_last']

        x,y = self.input_dims[:2]# technically switched
        for i in range(len(self.hs)):
            print("shape", self.input_dims, x, y)
            x = int(( x + 2 * self.padding - (self.kernel - 1) - 1) / self.stride + 1)
            y = int(( y + 2 * self.padding - (self.kernel - 1) - 1) / self.stride + 1)
        last_num = self.hs[-1] if not include_last else self.output_dim
        self.reduce_size = x * y * last_num 

        if len(self.hs) == 0:
            layers = [nn.Conv2d(self.channels, self.output_dim, self.kernel, self.stride, self.padding)]
        else:
            if len(self.hs) == 1:
                layers = [nn.Conv2d(self.channels, self.hs[0], self.kernel, self.stride, self.padding)]
            else:
                layers = ([nn.Conv2d(self.channels, self.hs[0], self.kernel, self.stride, self.padding), nn.ReLU(inplace=True)] + 
                      sum([[nn.Conv2d(self.hs[i-1], self.hs[i], self.kernel, self.stride, self.padding), nn.ReLU(inplace=True)] for i in range(1, len(self.hs) - 1)], list())
                      + [nn.Conv2d(self.hs[-2], self.hs[-1], self.kernel, self.stride, self.padding), nn.ReLU(inplace=True)])
            if include_last: # if we include last, we need a relu after second to last. If we do not include last, we assume that there is a layer afterwards so we need a relu after the second to last
                layers += [nn.Conv2d(self.hs[-1], self.output_dim, self.kernel, self.stride, self.padding)]
        self.conv = nn.Sequential(*layers)
        self.model = nn.ModuleList([self.conv])
        self.final = None
        if self.reduce:
            self.final = nn.Linear(self.reduce_size, self.output_dim)
            self.model = nn.ModuleList([self.conv] + [self.final])
        self.train()
        self.reset_parameters()

    def forward(self, x):
        x = self.conv(x)
        if self.reduce:
            x = x.reshape(-1, self.reduce_size)
            x = self.final(x)
        return x

class PairConvNetwork(Network):
    def __init__(self, **kwargs): 
        super().__init__(**kwargs)
        self.first_dim = kwargs["first_obj_dim"]
        self.input_dims = kwargs["input_dims"]
        self.hidden_sizes = kwargs["hidden_sizes"]
        self.output_dim = kwargs["output_dim"]
        kwargs["num_inputs"] = self.num_inputs - self.first_dim
        kwargs["output_dim"] = self.hidden_sizes[-1] # last hidden size is the output size of the conv network
        kwargs["hidden_sizes"] = self.hidden_sizes[:-1]
        redu = kwargs["reduce"] if "reduce" in kwargs else False
        kwargs["include_last"] = True
        kwargs["reduce"] = True
        self.conv = Basic2DConvNetwork(**kwargs)
        kwargs["num_inputs"] = self.first_dim + self.hidden_sizes[-1]
        kwargs["output_dim"] = self.output_dim
        kwargs["hidden_sizes"] = [128,128]
        self.final_layer = BasicMLPNetwork(**kwargs)
        self.model = nn.ModuleList([self.conv, self.final_layer])

        kwargs["reduce"] = redu

        self.train()
        self.reset_parameters()

    def forward(self, x):
        px = x[...,:self.first_dim]
        if len(px.shape) <= 1:
            px = px.reshape(-1, *px.shape)
        x = x[...,self.first_dim:]
        x = x.reshape(-1, *self.input_dims)
        x = x.transpose(2,3).transpose(1,2)
        x = self.conv(x)
        x = torch.cat([x, px], dim=-1)
        x = self.final_layer(x)
        return x

class PointNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # assumes the input is flattened list of input space sized values
        # needs an object dim
        self.object_dim = kwargs['object_dim']
        self.hs = kwargs["hidden_sizes"]
        self.aggregate_final = kwargs["aggregate_final"]
        self.output_dim = kwargs["output_dim"]
        kwargs["include_last"] = True
        self.conv = BasicConvNetwork(**kwargs)
        if kwargs["aggregate_final"]:
            kwargs["include_last"] = True
            kwargs["num_inputs"] = self.output_dim
            kwargs["hidden_sizes"] = list() # TODO: hardcoded final hidden sizes for now
            self.MLP = BasicMLPNetwork(**kwargs)
            self.model = nn.Sequential(self.conv, self.MLP)
        else:
            self.model = [self.conv]
        self.train()
        self.reset_parameters()

    def forward(self, x):
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        nobj = x.shape[-1] // self.object_dim
        x = x.view(-1, nobj, self.object_dim).transpose(1,2)
        # print(x.shape, self.aggregate_final, x)
        x = self.conv(x).transpose(2,1)
        # TODO: could use additive instead of max
        if self.aggregate_final:
            x = torch.max(x, 2, keepdim=True)[0]
            x = x.view(-1, self.output_dim)
            x = self.MLP(x)
            # print(x)
            # print(x.sum(dim=0))
            # error

        return x

class PairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # assumes the input is flattened list of input space sized values
        # needs an object dim
        # does NOT require a fixed number of objects, because it collects through a max operator
        self.object_dim = kwargs['object_dim']
        self.hs = kwargs["hidden_sizes"]
        self.first_obj_dim = kwargs["first_obj_dim"]
        self.post_dim = kwargs['post_dim']
        self.drop_first = kwargs['drop_first'] if 'drop_first' in kwargs else False
        if kwargs["first_obj_dim"] > 0: # only supports one to many concatenation, not many to many
            if not self.drop_first:
                kwargs["object_dim"] += self.first_obj_dim
        self.conv_dim = self.hs[-1] if len(self.hs) > 0 else kwargs['output_dim']
        if kwargs["aggregate_final"] and kwargs['post_dim'] > 0:
            self.output_dim = self.hs[-1] * 2
            kwargs["include_last"] = False
        elif kwargs["aggregate_final"] and kwargs['post_dim'] <= 0: 
            self.output_dim = self.hs[-1]
            kwargs["include_last"] = False
        else:
            kwargs["include_last"] = True
            self.output_dim = kwargs['output_dim']
        kwargs['output_dim'] = self.output_dim
        layers = list()
        print("object, first object", kwargs['object_dim'], kwargs['first_obj_dim'])
        self.conv = BasicConvNetwork(**kwargs)
        layers.append(self.conv)
        if kwargs['post_dim'] > 0:
            kwargs["num_inputs"] = kwargs['post_dim'] + kwargs['first_obj_dim']
            kwargs["num_outputs"] = self.hs[-1]
            self.post_channel = BasicMLPNetwork(**kwargs)
            layers.append(self.post_channel)
        self.aggregate_final = kwargs["aggregate_final"]
        self.activation_final = kwargs["activation_final"] if "activation_final" in kwargs else ""
        self.softmax = nn.Softmax(-1)
        if kwargs["aggregate_final"]: # does not work with a post-channel
            kwargs["include_last"] = True
            kwargs["num_inputs"] = self.output_dim
            kwargs["num_outputs"] = self.num_outputs
            kwargs["hidden_sizes"] = [256] # TODO: hardcoded final hidden sizes for now
            self.MLP = BasicMLPNetwork(**kwargs)
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
        # print(self.first_obj_dim, self.object_dim)
        nobj = x.shape[-1] // self.object_dim
        # reshape the object components
        x = x.view(-1, nobj, self.object_dim)
        if self.first_obj_dim > 0 and not self.drop_first:
            # append the pre components to every object and reshape
            broadcast_fx = torch.stack([fx.clone() for i in range(nobj)], dim=len(fx.shape) - 1)
            x = torch.cat((broadcast_fx, x), dim=-1)
        # transpose because conv-nets have reversed dimensions
        x = x.transpose(-1,-2)
        return x, fx, px, batch_size

    def run_networks(self, x, px, batch_size):
        # print(x.shape)
        x = self.conv(x)
        if self.aggregate_final:
            # TODO: could use additive instead of max
            # print(x.shape)
            # x = torch.max(x, 2, keepdim=True)[0]
            x = torch.mean(x, 2)
            # print(x.shape)
            x = x.view(-1, self.conv_dim)
            # print(x.shape)
            if self.post_dim > 0:
                px = self.post_channel(px)
                x = torch.cat([x,px], dim=-1)
            x = self.MLP(x)
            # print("post mlp", x.shape)
        else:
            x = x.transpose(2,1)
            x = x.reshape(batch_size, -1)
        if len(self.activation_final) > 0:
            if self.activation_final == "sigmoid":
                x = torch.sigmoid(x)
            elif self.activation_final == "tanh":
                x = torch.tanh(x)
            elif self.activation_final == "softmax":
                x = self.softmax(x)
                # print("tanh", x)
        return x


    def forward(self, x):
        x, fx, px, batch_size = self.slice_input(x)
        x = self.run_networks(x, px, batch_size)
        return x

class TransformerPairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregate_final = kwargs["aggregate_final"]
        self.object_dim = kwargs["object_dim"]
        self.output_dim = kwargs["output_dim"]
        self.hidden_sizes = kwargs["hidden_sizes"]
        # requires a fixed number of objects for regular pairwise sizes

        kwargs["aggregate_final"] = False
        kwargs["hidden_sizes"] = self.hidden_sizes[:len(self.hidden_sizes) - 1]
        kwargs["output_dim"] = self.hidden_sizes[-1]
        self.key_dim = self.hidden_sizes[-1]
        self.key_network = PairNetwork(**kwargs)
        kwargs["drop_first"] = True
        self.query_network = PairNetwork(**kwargs)
        kwargs["drop_first"] = False
        
        self.value_network = PairNetwork(**kwargs)
        self.softmax =  nn.Softmax(-1)
        
        self.layers = [self.key_network, self.value_network, self.query_network, self.softmax]

        kwargs["aggregate_final"] = self.aggregate_final
        kwargs["object_dim"] = self.hidden_sizes[-1]
        kwargs["output_dim"] = self.output_dim
        kwargs["hidden_sizes"] = [128, 128] # hardcoded final layer
        self.output_network = PointNetwork(**kwargs)
        self.layers.append(self.output_network)

        # reset kwargs
        kwargs["object_dim"] = self.object_dim
        kwargs["hidden_sizes"] = self.hidden_sizes
        self.model = nn.ModuleList(self.layers)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        kv_x, fx, px, batch_size = self.key_network.slice_input(x)
        K = self.key_network.run_networks(kv_x, px, batch_size).reshape(batch_size, -1, self.key_dim) # B x nobj x d
        V = self.value_network.run_networks(kv_x, px, batch_size).reshape(batch_size, -1, self.key_dim)
        q_x, fx, px, batch_size = self.query_network.slice_input(x)
        Q = self.query_network.run_networks(q_x, px, batch_size).reshape(batch_size, -1, self.key_dim)
        # print("KVQ", K.shape, V.shape, Q.shape)
        alpha = self.softmax(torch.matmul(Q, K.transpose(2,1)) / np.sqrt(self.key_dim)) # B x nobj x nobj keys along rows dim -1 
        # print(alpha.sum(), alpha.abs().sum())
        Z = torch.matmul(alpha, V) # B x nobj x d
        # TODO: pairwise information in a single dimension
        # print("zhape", Z.shape)
        x = Z.reshape(batch_size, -1)
        # print(x.abs().sum(), x.shape)
        x = self.output_network(x)
        # print(x.abs().sum(), x.sum())
        return x

class MultiheadedTransformerPairNetwork(Network):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.aggregate_final = kwargs["aggregate_final"]
        self.object_dim = kwargs["object_dim"]
        self.output_dim = kwargs["output_dim"]
        self.hidden_sizes = kwargs["hidden_sizes"]
        # requires a fixed number of objects for regular pairwise sizes
        self.num_heads = kwargs["num_heads"]

        kwargs["hidden_sizes"] = self.hidden_sizes[:len(self.hidden_sizes) - 1]
        kwargs["output_dim"] = kwargs["hidden_sizes"][-1]
        self.heads = list()
        for h in range(self.num_heads):
            self.heads.append(TransformerPairNetwork(**kwargs))

        kwargs["object_dim"] = self.hidden_sizes[-1] * self.num_heads
        kwargs["hidden_sizes"] = [128, 128, 128] # hardcoded final layer
        kwargs["output_dim"] = self.output_dim
        self.output_network = PointNetwork(**kwargs)
        self.layers = self.heads + [self.output_network]
        
        # reset kwargs
        kwargs["object_dim"] = self.object_dim
        kwargs["hidden_sizes"] = self.hidden_sizes
        self.model = nn.ModuleList(self.layers)
        self.train()
        self.reset_parameters()

    def forward(self, x):
        outputs = list()
        batch_size = x.shape[0] if len(x.shape) > 1 else 1
        for head in self.heads:
            outputs.append(head(x))
        outputs = torch.cat(outputs, dim=-1)
        result = self.output_network(outputs)
        if self.aggregate_final:
            return result
        else:
            return result.reshape(batch_size, -1)


class FactoredMLPNetwork(Network):    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.factor = kwargs['factor']
        self.num_layers = kwargs['num_layers']
        self.use_layer_norm = kwargs['use_layer_norm']
        self.MLP = BasicMLPNetwork(**kwargs)
        self.train()
        self.reset_parameters()

    def basic_operations(self, x):
        # add, subtract, outer product
        return

    def forward(self, x):
        x = self.basic_operations(x)
        x = self.MLP(x)
            # print(x)
            # print(x.sum(dim=0))
            # error

        return x