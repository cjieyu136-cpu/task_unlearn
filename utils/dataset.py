import torch
from torch.utils.data import Dataset
from os import listdir
import pandas as pd
import numpy as np
from .utils import set_random_seed
from pypower.idx_bus import PD
from pypower.idx_brch import RATE_A, RATE_B, RATE_C
import pypower
from pypower import api as pypower_api
from omegaconf import OmegaConf

def case_modifier(case_config):
    """
    return the modified pypower.api case
    """
    case = getattr(pypower_api, case_config.name)()

    # rescale the load
    pd = case['bus'][:, PD] * case_config["load_scale"]
    # add load to the zero-load bus
    average_pd = np.mean(pd)
    max_pd = np.max(pd)
    for i in range(len(pd)):
        if pd[i] == 0:
            pd[i] = np.min([average_pd * np.sqrt(i + 1), max_pd * 0.8]) # no randomness here
    
    # modify the case file
    case['bus'][:,PD] = pd
    pf_max = np.array(OmegaConf.to_object(case_config["RATE_A"])) if case_config["RATE_A"] != None else case['branch'][:,RATE_A]
    case['branch'][:,RATE_A] = pf_max
    case['branch'][:,RATE_B] = pf_max  # should make no impact
    case['branch'][:,RATE_C] = pf_max  # should make no impact

    # load shedding > over-generation > first order
    case["second_coeff"] = OmegaConf.to_object(case_config["second_coeff"])  # cost of second order in $/p.u.^2
    case["first_coeff"] = OmegaConf.to_object(case_config["first_coeff"])  # cost of first order in $/p.u.
    max_first_coeff = np.max(case["first_coeff"])
    max_second_coeff = np.max(case["second_coeff"])
    case['load_shed_coeff'] = max_first_coeff * case_config["load_shed_coeff_scale"] # cost of load shedding in $/MW  (very large)
    case['gen_storage_coeff'] =  max_first_coeff * case_config["gen_storage_coeff_scale"] # cost of over-generation
    case['load_shed_coeff_second'] =   max_second_coeff * case_config["load_shed_coeff_scale"] # cost of load shedding in $/MW^2
    case['gen_storage_coeff_second'] =  max_second_coeff * case_config["gen_storage_coeff_scale"] # cost of over-generation in $/MW^2
    
    if case_config["shunt_tap_linecharge"] == False:
        case['bus'][:, 4:6] = 0
        case['branch'][:, 4] = 0
        case['branch'][:, 8:10] = 0
    
    return case

class MyDataset(Dataset):
    
    def __init__(self, configs, mode):
        
        """
        mode: 'train' or 'test'
        """

        case_config = configs.case
        model_config = configs.model
        
        case = case_modifier(case_config)              # modified pyppower.api case
        load_standard = case["bus"][:,PD]   
        load_standard = load_standard[load_standard > 0]
        self.no_load = len(load_standard)
        
        is_scale = model_config["is_scale"]
        self.exp_dim = model_config["exp_dim"]
        data_prop = configs.data["data_prop"]
        train_prop = configs.data["train_prop"]
        
        # ! control the random seed here 
        set_random_seed(configs.data["random_seed"])
        
        """
        clean dataset
        """
        DATA_DIR = f'data/data_case{case_config.no_bus}/'
        FILE_NAME = listdir(DATA_DIR)
        FILE_NAME = sorted(FILE_NAME, key = lambda x: int(x.split('_')[1].split('.')[0]))[:self.no_load]
        # print(FILE_NAME)
        
        TRAIN_DATA = []
        for file_name in FILE_NAME:
            TRAIN_DATA.append(pd.read_csv(DATA_DIR + file_name).values)
        
        # target (ground truth load)
        load = []
        for i in range(len(FILE_NAME)):
            load.append(TRAIN_DATA[i][:, -1])
        
        target = np.array(load).T  # (sample, no_bus)
        # rescale the load by the standard load, so that the max load equals to standard load, power flow is feasible
        target = target / target.max(axis = 0)[None, :] * load_standard[None, :]
        assert np.allclose(target.max(axis = 0), load_standard, atol = 1e-2)
        
        target = target / case["baseMVA"] # in p.u.
        
        # feature
        NO_SAMPLE = len(target)
        NO_FEATURE = len(TRAIN_DATA[0][0]) - 1

        feature = np.zeros((NO_SAMPLE * len(FILE_NAME), NO_FEATURE))

        for i in range(len(FILE_NAME)):
            position_index = np.arange(i, NO_SAMPLE * len(FILE_NAME), len(FILE_NAME))
            feature[position_index] = TRAIN_DATA[i][:, :-1]
        
        feature = feature.reshape(NO_SAMPLE, len(FILE_NAME), NO_FEATURE)  # (sample, no_bus, no_feature)
    
        # scale
        # ! only scale the target, the feature has already been scaled
        self.target_mean = target.mean(axis = 0)
        self.target_std = target.std(axis = 0)
        
        self.is_scale = False
        if is_scale:
            self.is_scale = True    
            target = (target - self.target_mean) / self.target_std
        
        # data proportion: samller data size for test
        if data_prop != 1:
            # random sample part of the data
            data_size = int(data_prop * len(target))
            data_index = np.sort(np.random.choice(len(target), data_size, replace = False))
            feature = feature[data_index]
            target = target[data_index]
        
        if configs.data.random_split:
            # random split the train test dataset
            # because the data is one year, so for each month, the first part is train and the last part is test
            train_size = int(train_prop * len(target))
            test_size = len(target) - train_size

            train_index =  np.random.choice(len(target), train_size, replace = False)
            test_index = np.array(list(set(np.arange(len(target))) - set(train_index)))
            
            one_month_len = int(len(target) / 12)
            test_index = []
            for i in range(12):
                month_index = np.arange(i * one_month_len, (i + 1) * one_month_len)
                test_index.append(month_index[-int(test_size/12):])
            
            test_index = np.concatenate(test_index, axis = 0)
            train_index = np.array(list(set(np.arange(len(target))) - set(test_index)))
            
        else:
            # not random sampling
            train_size = int(train_prop * len(target))
            test_size = len(target) - train_size

            test_index = np.arange(test_size)                  # the first part is test
            train_index = np.arange(test_size, len(target))    # the last part is train
        
        # add the channel dimension for use conv layer
        if self.exp_dim:
            feature = feature[:,None,:,:]
            
        # ! currently disable the unlearn dataset
        """
        output dataset
        """
        if mode == 'train':
            # assert self.core_prop == 0, "core_prop should be 0 when mode is train"
            target = target[train_index]
            feature = feature[train_index]
        elif mode == 'test':
            target = target[test_index]
            feature = feature[test_index]
    
        self.feature = torch.from_numpy(feature).float()
        self.target = torch.from_numpy(target).float()
        self.target_mean = torch.from_numpy(self.target_mean).float()
        self.target_std = torch.from_numpy(self.target_std).float()

    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        
        return self.feature[idx], self.target[idx]

class NewDataset(Dataset):
    
    def __init__(self, feature, target, mean = 0, std = 1):
        self.feature = feature
        self.target = target
        self.target_mean = torch.tensor(mean) if type(mean) != torch.Tensor else mean
        self.target_std = torch.tensor(std) if type(std) != torch.Tensor else std
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.feature[idx], self.target[idx]


class DatasetWithWeight(Dataset):
    """
    [0]: feature
    [1]: target
    [2]: weight
    """
    
    def __init__(self, dataset_ori, weight):
        self.feature = dataset_ori.feature
        self.target =  dataset_ori.target
        self.weight = torch.from_numpy(weight).float() if type(weight) != torch.Tensor else weight
        self.target_mean = dataset_ori.target_mean
        self.target_std = dataset_ori.target_std
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.feature[idx], self.target[idx], self.weight[idx]


def return_dataset(configs):
    
    dataset_train = MyDataset(configs =configs, mode = 'train')
    dataset_test = MyDataset(configs =configs, mode = 'test')
    
    return dataset_train, dataset_test