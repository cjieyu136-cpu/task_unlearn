# construct a simple convnet for load forecast
# input: (batch_size, 1, no_bus, no_feature)
# output: (batch_size, no_bus)

from torch import nn
from cvxpylayers.torch import CvxpyLayer
import torch
from functools import partial
from einops.layers.torch import Rearrange, Reduce

class Flatten(nn.Module):
    def forward(self, x):
        return x.reshape(x.size(0), -1)

class NN_CONV(nn.Module):
    """
    convolutional neural network for load forecast with 2 conv layers and 2 linear layers
    first train the model on the core dataset, then train the last layer (linear) on the sensitive dataset
    """
    
    def __init__(self, in_ch, input_length, input_width, width, linear_size, output_size):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, 4*width, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(4*width, 4*width, 4, stride=2, padding=1)
        self.flatten = Flatten()
        self.lin1 = nn.Linear(4*width*int(input_length/2)*int(input_width/2), linear_size)
        self.lin2 = nn.Linear(linear_size,output_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.lin1(x) 
        feature = self.tanh(x)         # output of the extractor
        forecast = self.lin2(feature)  # output of the predictor (this is a linear model)
        return feature, forecast

"""
functions for mlp-mixer
code from https://github.com/lucidrains/mlp-mixer-pytorch/blob/main/mlp_mixer_pytorch/mlp_mixer_pytorch.py
"""

pair = lambda x: x if isinstance(x, tuple) else [x, x]

class PreNormResidual(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        return self.fn(self.norm(x)) + x

def FeedForward(dim, expansion_factor = 4, dropout = 0., dense = nn.Linear):
    inner_dim = int(dim * expansion_factor)
    return nn.Sequential(
        dense(dim, inner_dim),
        nn.GELU(),
        nn.Dropout(dropout),
        dense(inner_dim, dim),
        nn.Dropout(dropout)
    )

def MLPMixer(*, image_size, channels, patch_size, dim, depth, num_classes, expansion_factor = 1, expansion_factor_token = 0.5, dropout = 0.):
    image_h, image_w = pair(image_size)
    assert (image_h % patch_size) == 0 and (image_w % patch_size) == 0, 'image must be divisible by patch size'
    num_patches = (image_h // patch_size) * (image_w // patch_size)
    chan_first, chan_last = partial(nn.Conv1d, kernel_size = 1), nn.Linear

    class NN_MIXER(nn.Module):
        def __init__(self):

            super().__init__()

            self.layers = nn.Sequential()
            self.layers.add_module('patching', Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1 = patch_size, p2 = patch_size))
            self.layers.add_module('linear', nn.Linear((patch_size ** 2) * channels, dim))
            for i in range(depth):
                self.layers.add_module(f'token_{i}', PreNormResidual(dim, FeedForward(num_patches, expansion_factor, dropout, chan_first)))
                self.layers.add_module(f'channel_{i}', PreNormResidual(dim, FeedForward(dim, expansion_factor_token, dropout, chan_last)))
            
            self.layers.add_module('layer_norm', nn.LayerNorm(dim))
            self.layers.add_module('pooling', Reduce('b n c -> b c', 'mean'))
            self.layers.add_module('lin1', nn.Linear(dim, dim))
            self.layers.add_module('tanh', nn.ReLU())
            self.layers.add_module('lin2', nn.Linear(dim, num_classes))
        
        def forward(self, x):
            for layer in self.layers[:-1]:
                x = layer(x)  
            forecast = self.layers[-1](x)

            return x, forecast
    
    return NN_MIXER()


class Stage_One_Layer(torch.nn.Module):
    # NOTE: we do not need to normalize the load forecast here
    def __init__(self, prob):
        """
        a torch layer for convex optimization problem
        prob: a cvxpy problem parameterized by load forecast
        """
        super().__init__()
        self.layer = CvxpyLayer(prob, 
                parameters = prob.parameters(), 
                variables = prob.variables()
                )
        
    def forward(self, load_forecast):
        # TODO: not sure how the parameter is listed in order?
        return self.layer(load_forecast)[0]

class Stage_Two_Layer(torch.nn.Module):
    def __init__(self, prob):
        super().__init__()
        self.layer = CvxpyLayer(prob,
                parameters = prob.parameters(),
                variables = prob.variables()
                )
    
    def forward(self, pg, load_true):
        # load_shedding, generator_storage
        # TODO: not sure how the parameter is listed in order?
        outputs = self.layer(pg, load_true)
        # return outputs[0], outputs[1]
        return outputs[0], outputs[1]

class SPO(nn.Module):
    """
    concate the neural network with the stage one and stage two differentiable layers
    """
    def __init__(self, trained_model, operator, mean, std):
        super().__init__()
        self.nn_model = trained_model
        self.ReLU = nn.ReLU()
        self.stage_one_layer = Stage_One_Layer(operator.prob_1)
        self.stage_two_layer = Stage_Two_Layer(operator.prob_2)
        self.mean = torch.tensor(mean) if type(mean) != torch.Tensor else mean
        self.std = torch.tensor(std) if type(std) != torch.Tensor else std
    
    def forward(self, feature, target):
        # ! if the original data is scaled, we need to unscale it before pass it to the model
        nn_output = self.nn_model(feature) # (batch_size, no_bus) 
        forecast_load = nn_output * self.std + self.mean
        target = target * self.std + self.mean
        # stage one
        pg = self.stage_one_layer(forecast_load)
        
        # stage two
        ls, gs = self.stage_two_layer(pg, target)
        
        # return nn_output, pg, ls, gs: nn_output is scaled; pg, ls, gs are not scaled
        # concatenate the output to allow the backpropagation
        return torch.cat((nn_output, pg, ls, gs), axis = 1)    