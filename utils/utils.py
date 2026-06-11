import torch
import numpy as np
import random
def set_random_seed(random_seed):
    torch.random.manual_seed(random_seed)
    np.random.seed(random_seed)
    random.seed(random_seed)