"""
high level functions which calls functions in utils/
"""

from utils.funcs import AnalyticSolverAffine, ModelAffine, AnalyticSolverNNAffine, ModelNNAffine
import torch
import numpy as np
from utils.net import NN_CONV, SPO, MLPMixer
import os
from utils.optimization import Operator
from torch_influence.base import BaseObjective
from torch_influence.modules import AutogradInfluenceModule, CGInfluenceModule
import torch.nn.functional as F
from omegaconf import OmegaConf


from utils import set_random_seed, NewDataset

def return_affine_model(dataset_train):
    """
    train an affine model on the train dataset
    """
    analytic_solver = AnalyticSolverAffine(dataset_collect = {'train': dataset_train})
    parameter_analytic = analytic_solver.fit_regressor(mode = 'train')
    return parameter_analytic

def return_nn_model(config, is_load, dataset = None):
    """
    load the trained nn model
    dataset: all or core
    """

    if config.model.type == 'nn_conv':
        no_load = config.data.no_load
        in_ch = config.model.in_ch
        input_width = config.model.input_width
        width = config.model.width
        linear_size = config.model.linear_size
        net = NN_CONV(in_ch = in_ch, input_length=no_load, input_width = input_width, 
                    width = width, linear_size = linear_size, output_size = no_load)
    elif config.model.type == 'nn_mixer':
        net = MLPMixer(
            image_size = (config.model.image_height, config.model.image_width),
            channels = config.model.in_ch,
            patch_size = config.model.patch_size,
            dim = config.model.linear_size,
            depth = config.model.depth,
            num_classes = config.model.image_height
        )
    
    if is_load:
        save_dir = config.model.save_dir + f'/{dataset}.pth'
        print(f'Loading model from {save_dir}')
        assert os.path.exists(save_dir), 'the nn model file does not exist'
        net.load_state_dict(torch.load(save_dir))
    
    return net

def return_nn_affine_model(dataset_train):
    
    analytic_solver = AnalyticSolverNNAffine(dataset_collect = {'train': dataset_train})
    parameter_analytic = analytic_solver.fit_regressor(mode = 'train')
    return parameter_analytic
    
def return_trained_model(config, model_type, dataset_train, is_spo, dataset = None):
    """
    return the trained load forecast model as **pytorch** module
    model_type:
        1. affine: generate the model here
        2. nn: load the trained model
    is_spo: if True, return the SPO model
            ! carefully check the mean and std
    """
    
    if model_type == 'linear':
        # train linear model
        parameter_analytic = return_affine_model(dataset_train)
        # convert linear model into pytorch module
        model = ModelAffine(parameter_analytic)
    
    elif model_type == 'nn_conv' or model_type == 'nn_mixer':
        model = return_nn_model(config, is_load=True, dataset = dataset)
    
    elif model_type == 'nn_conv-affine' or model_type == 'nn_mixer-affine':
        parameter_analytic = return_nn_affine_model(dataset_train)
        model = ModelNNAffine(parameter_analytic)
        
    else:
        raise ValueError('model_type should be affine or nn')
    
    if is_spo:
        operator = Operator(case_config = config.case)
        if dataset_train.is_scale:
            mean = dataset_train.target_mean
            std = dataset_train.target_std
        else:
            mean = 0
            std = 1
        model = SPO(trained_model = model, operator = operator, mean = mean, std = std)
    
    model.eval()
    
    return model

def return_objective(loss_type_dict, is_scale, with_weight = False, **kwargs):
    """
    return the objective that defines the train loss and test loss to define the module
        train_loss: mse or mape
        test_loss: mse or mape or cost
    
    training loss by default taken to be train_loss_on_outputs + train_regularization
    
    output of spo model: (forecast_load, pg, ls, gs)
        1. for scaled load, forecast_load is scaled, pg, ls, gs are not scaled
        2. for unscaled load, all not scaled
        
    settings on the scaling:
        when train or test is mape, the scaled load needs to be unscaled    
    
        with_weight: if true, the last element of batch is the weight of each sample in the batch, 
                    this is used for reweighting the unlearn dataset
    """
    
    """
    Affine objective
    """
    
    class MSE_MSE(BaseObjective):
        """
        train loss: mse, test loss: mse
        we need to distinguish the scaled and unscaled data
        """ 
            
        def train_outputs(self, model, batch):
            # batch is a tuple of (feature, target)
            return model(batch[0])

        def train_loss_on_outputs(self, outputs, batch):
            if with_weight:
                loss = torch.mean(F.mse_loss(outputs, batch[1], reduction = 'none'), axis = 1)
                # print(loss.shape, batch[2].shape)
                return torch.mean(loss * batch[2])
            else:
                return F.mse_loss(outputs, batch[1])  # mean reduction required
                
        def train_regularization(self, params):
            # no regularization
            return 0. * torch.square(params.norm())

        def test_loss(self, model, params, batch):
            outputs = model(batch[0])
            if with_weight:
                loss = torch.mean(F.mse_loss(outputs, batch[1], reduction = 'none'), axis = 1)
                # print(loss.shape, batch[2].shape)
                return torch.mean(loss * batch[2])
            else:
                return F.mse_loss(outputs, batch[1])
            
            # return F.mse_loss(model(batch[0]), batch[1])
    
    class MSE_MAPE(BaseObjective):
        
        """
        if the model is trained on scaled data, then the test data should be unscaled
        """

        def __init__(self, mean = 0, std = 1):
            self.mean = torch.tensor(mean).float() if type(mean) != torch.Tensor else mean
            self.std = torch.tensor(std).float() if type(std) != torch.Tensor else std
        
        def train_outputs(self, model, batch):
            # batch is a tuple of (feature, target)
            return model(batch[0])
        
        def train_loss_on_outputs(self, outputs, batch):
            return F.mse_loss(outputs, batch[1])
        
        def train_regularization(self, params):
            # no regularization
            return 0. * torch.square(params.norm())
        
        def test_loss(self, model, params, batch):
            # unscale
            forecast_load = model(batch[0]) * self.std + self.mean
            target_load = batch[1] * self.std + self.mean
            return torch.mean(torch.abs(forecast_load - target_load) / target_load)
    
    class MSE_COST(BaseObjective):
        
        """
        the spo model already unscales the load forecast if applicable
        """
        
        def __init__(self, operator):
            # operator is used to solve the power system operation problem
            self.second_coeff = torch.tensor(operator.second_coeff).float()
            self.first_coeff = torch.tensor(operator.first_coeff).float()
            self.load_shed_coeff_second = torch.tensor(operator.load_shed_coeff_second).float()
            self.load_shed_coeff = torch.tensor(operator.load_shed_coeff).float()
            self.gen_storage_coeff_second = torch.tensor(operator.gen_storage_coeff_second).float()
            self.gen_storage_coeff = torch.tensor(operator.gen_storage_coeff).float()
        
        def train_outputs(self, model, batch):
            # ! output the forecast load as it is used during training
            output = model(batch[0], batch[1])
            forecast_load = output[:, :14]
            return forecast_load
        
        def train_loss_on_outputs(self, outputs, batch):
            # outputs is the forecast load
            return F.mse_loss(outputs, batch[1])  # mean reduction required

        def train_regularization(self, params):
            # no regularization
            return 0. * torch.square(params.norm())

        def test_loss(self, model, params, batch):
            # consider the cost
            # ! the size is (batch_size, 14 + 5 + 14 + 5)
            # todo: the size of the output is not flexible for other power system cases
            outputs = model(batch[0], batch[1]) 
            # forecast_load = outputs[:, :14]
            pg = outputs[:, 14:19]
            ls = outputs[:, 19:33]
            gs = outputs[:, 33:]
            
            # second_pg + first_pg + second_ls + first_ls + second_gs + first_gs
            loss_gen = torch.square(pg) @ self.second_coeff + pg @ self.first_coeff
            loss_ls = torch.square(ls).sum(axis = 1) * self.load_shed_coeff_second + ls.sum(axis=1) * self.load_shed_coeff
            loss_gs = torch.square(gs).sum(axis = 1) * self.gen_storage_coeff_second + gs.sum(axis=1) * self.gen_storage_coeff
            loss = loss_gen + loss_ls + loss_gs
            return torch.mean(loss)
    
    # only mse_mape requires mean and std        
    if loss_type_dict['train'] == 'mse':
        if loss_type_dict['test'] == 'mse':
            return MSE_MSE()
        elif loss_type_dict['test'] == 'mape':
            if is_scale:
                # unscale
                mean = kwargs['target_mean']
                std = kwargs['target_std']
                return MSE_MAPE(mean = mean, std = std)
            else:
                return MSE_MAPE()
        elif loss_type_dict['test'] == 'cost':
            return MSE_COST(kwargs['operator'])
        else:
            raise ValueError("when train loss is 'mse', test loss should be either 'mse' or 'mape' or 'cost'")
    else:
        raise ValueError("currently only support mse loss for training!")
    

def return_module(configs, loss_type_dict, loader_dict, model, method, 
                device = 'cpu', with_weight = False, watch_progress = False):
    """
    return the module defined by the torch-influence package
    
    loss_type_dict: with keys 'train': 'mse' or 'mape' and 'test': 'mse', 'mape', or 'cost' to define the loss
        {
            'train': 'mse' or 'mape',            # by default, the train loss is mse and we never consider cost for training
            'test': 'mse' or 'mape' or 'cost'    # we can evaluate the performance on the test dataset by its mse, mape, or generator cost
        }
    
    loader_dict: with keys 'train' and 'test' to define the dataset
        the hessian is calculated on the train dataset
        the grad can be both calculated on the train dataset and test dataset. we can use this to control unlearning from the remain or unlearn dataset
    
    model: neural network model (always to be a pytorch layer of linear model)
    
    model_type: 'nn', 'linear'
    
    method: direct or cg
    """
    cfg_model = configs['model']
    cfg_case = configs['case']

    damp = float(OmegaConf.select(configs, "damping", default=cfg_model['damp']))  # the regularization term for positive definiteness
    gnh = bool(OmegaConf.select(configs, "gnh", default=cfg_model['gnh']))    # whether to use generalized newton's method
    is_scale = cfg_model['is_scale']  # whether the model is trained on scaled data
    cg_tol = float(OmegaConf.select(configs, "cg_tol", default=cfg_model.get('cg_tol', 1e-5)))
    cg_maxiter = int(OmegaConf.select(configs, "cg_maxiter", default=cfg_model.get('cg_maxiter', 1000)))
    cg_atol = float(OmegaConf.select(configs, "cg_atol", default=cfg_model.get('cg_atol', 0.0)))
    
    # define the objective for using the influence function module
    kwargs = {}
    if loss_type_dict['test'] == 'cost':
        kwargs['operator'] = Operator(case_config = cfg_case)
    
    elif is_scale and loss_type_dict['train'] == 'mse' and loss_type_dict['test'] == 'mape':
        # this is the only case we need to unscale the ouptut of the model
        kwargs['target_mean'] = loader_dict['train'].dataset.target_mean
        kwargs['target_std'] = loader_dict['train'].dataset.target_std

    myobjective = return_objective(loss_type_dict = loss_type_dict, 
                                is_scale = is_scale, with_weight=with_weight, 
                                **kwargs)

    if method == 'direct':
        module = AutogradInfluenceModule(
            model=model,
            objective=myobjective,  
                train_loader=loader_dict['train'], # for exact unlearning, we need to calculate the hessian on the remain dataset
                test_loader=loader_dict['test'],   # this can be replaced by unlearn_loader which is also exact
                device=device,
                damp=damp,
                check_eigvals = True
            )
    
    elif method == 'cg':
        module = CGInfluenceModule(
            model=model,
            objective=myobjective,  
                train_loader=loader_dict['train'], # for exact unlearning, we need to calculate the hessian on the remain dataset
                test_loader=loader_dict['test'],  # this can be replaced by unlearn_loader which is also exact
                device=device,
                damp=damp,
                gnh = gnh,
                # settings for conjugate gradient
                watch_progress=watch_progress,
                tol = cg_tol,
                atol = cg_atol,
                maxiter = cg_maxiter,
            )
    else:
        raise ValueError("method should be either 'direct' or 'cg'")
    
    return module



def return_unlearn_datasets(influences, unlearn_prop, dataset_to_be_unlearn, mode, config):
    """
    return the unlearning dataset as the subset of the dataset_to_be_unlearn, 
    you can choose the mode of how to choose the unlearning dataset
    
    mode: 'helpful' or 'harmful' or 'random'

    influence: None or an array of influeces on the train dataset when each of the sample is unlearnt
            must not be None if mode is helpful or harmful
    find the samples in the train dataset which are helpful or harmful to the test dataset's mape loss
    """
    
    if mode == 'helpful' or mode == 'harmful':
        assert len(influences) == len(dataset_to_be_unlearn), "the length of influences should be the same as the dataset"

    set_random_seed(config.data.random_seed)
    
    # ! assume the maximum unlearning ratio is 0.3
    unlearn_no = int(unlearn_prop * len(dataset_to_be_unlearn))
    candidate_no = int(0.31 * len(dataset_to_be_unlearn)) 
    
    if mode == 'random':
        # randomly unlearn from the train dataset
        unlearn_index = np.random.choice(len(dataset_to_be_unlearn), unlearn_no, replace = False)
    elif mode == 'helpful':
        # find the samples that is most helpful to the test dataset performance
        unlearn_index = torch.argsort(influences, descending = True)[:candidate_no].numpy()
        print('ave. performance change of unlearning (positive for helpful): {}'.format(np.sum(influences.numpy()[unlearn_index])))
    elif mode == 'harmful':
        # find the samples that is most harmful to the test dataset performance
        unlearn_index = torch.argsort(influences, descending = False)[:candidate_no].numpy()  
    else:
        print('mode should be random or helpful or harmful')
        
    # randomly choose from the over_scale dataset
    unlearn_index = np.random.choice(unlearn_index, unlearn_no, replace = False)
    remain_index = [i for i in range(len(dataset_to_be_unlearn)) if i not in unlearn_index]
    assert len(set(remain_index).intersection(set(unlearn_index))) == 0, "the two sets should be disjoint"
    
    dataset_unlearn = NewDataset(dataset_to_be_unlearn.feature[unlearn_index], dataset_to_be_unlearn.target[unlearn_index], 
                                mean = dataset_to_be_unlearn.target_mean, std = dataset_to_be_unlearn.target_std)
    dataset_remain = NewDataset(dataset_to_be_unlearn.feature[remain_index], dataset_to_be_unlearn.target[remain_index], 
                                mean = dataset_to_be_unlearn.target_mean, std = dataset_to_be_unlearn.target_std)
    dataset_unlearn.is_scale = dataset_to_be_unlearn.is_scale
    dataset_remain.is_scale = dataset_to_be_unlearn.is_scale
    
    return dataset_unlearn, dataset_remain, unlearn_index, remain_index


def return_core_datasets(config, dataset_to_be_split):
    """
    return the core dataset and the sensitive dataset: only for nn model
    """
    
    core_prop = config.model['core_prop']
    is_random = config.model['is_random_core']
    set_random_seed(config.data.random_seed)    
    core_no = int(core_prop * len(dataset_to_be_split))
    
    if is_random:
        core_index = np.random.choice(len(dataset_to_be_split), core_no, replace = False)
    else:
        core_index = range(len(dataset_to_be_split))[:core_no]
    
    sensitive_index = [i for i in range(len(dataset_to_be_split)) if i not in core_index]
    assert len(set(sensitive_index).intersection(set(core_index))) == 0, "the two sets should be disjoint"
    mean = dataset_to_be_split.target_mean
    std = dataset_to_be_split.target_std
    dataset_core = NewDataset(dataset_to_be_split.feature[core_index], dataset_to_be_split.target[core_index], 
                            mean = mean, std = std)
    dataset_sensitive = NewDataset(dataset_to_be_split.feature[sensitive_index], dataset_to_be_split.target[sensitive_index], 
                                mean = mean, std = std)
    dataset_core.is_scale = dataset_to_be_split.is_scale
    dataset_sensitive.is_scale = dataset_to_be_split.is_scale
    
    return dataset_core, dataset_sensitive


def return_dataset_for_nn_affine(config, dataset_sensitive, dataset_test):
    """
    return the dataset used to train the linear model upon the nn model trained by the core dataset
    the linear model takes the output of the nn model (trained on the core dataset) as input
    """
    # nn trained on the core dataset
    nn_model_core = return_nn_model(config, is_load=True, dataset="core")
    nn_model_core.eval()
    
    with torch.no_grad():
        # find the output of the nn model on the core dataset
        feature_sensitive = nn_model_core(dataset_sensitive.feature)[0] # 0 represents the output of the feature extractor
        feature_test = nn_model_core(dataset_test.feature)[0]
    
    target_sensitive = dataset_sensitive.target
    target_test = dataset_test.target
    
    mean = dataset_sensitive.target_mean
    std = dataset_sensitive.target_std
    
    dataset_train = NewDataset(feature_sensitive, target_sensitive, mean, std)
    dataset_test = NewDataset(feature_test, target_test, mean, std)
    
    dataset_train.is_scale = dataset_sensitive.is_scale
    dataset_test.is_scale = dataset_sensitive.is_scale
    
    return dataset_train, dataset_test
