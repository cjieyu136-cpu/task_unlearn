# linear regression for load forecasting

from sklearn.linear_model import LinearRegression
from torch import nn
import torch
import numpy as np
import torch.nn.functional as F
import sys
sys.path.append(".")
from torch_influence import BaseObjective
from torch_influence import AutogradInfluenceModule, CGInfluenceModule

class SklearnSolver:
    # solving the linear regression problem using sklearn
    def __init__(self, dataset_train, dataset_test, dataset_remain, dataset_unlearn):
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_remain = dataset_remain
        self.dataset_unlearn = dataset_unlearn
    
    def select_dataset(self,mode):
        if mode == 'train':
            dataset = self.dataset_train
        elif mode == 'test':
            dataset = self.dataset_test
        elif mode == 'remain':
            dataset = self.dataset_remain
        elif mode == 'unlearn':
            dataset = self.dataset_unlearn
        else:
            raise ValueError("mode should be one of 'train', 'test', 'remain', 'unlearn'")
        
        self.dataset = dataset
        
        assert dataset.is_scale == False, "dataset should be unscaled"
    
    def fit_regressor(self, mode):
        self.select_dataset(mode)
        feature = self.dataset.feature.numpy()
        target = self.dataset.target.numpy()
        reg = LinearRegression().fit(feature, target)
        return reg

class AnalyticSolver:
    # solving the linear regression problem using the analytic solution
    def __init__(self, dataset_train, dataset_test, dataset_remain, dataset_unlearn):
        
        self.dataset_train = dataset_train
        self.dataset_test = dataset_test
        self.dataset_remain = dataset_remain
        self.dataset_unlearn = dataset_unlearn
    
    def select_dataset(self,mode):
        if mode == 'train':
            dataset = self.dataset_train
        elif mode == 'test':
            dataset = self.dataset_test
        elif mode == 'remain':
            dataset = self.dataset_remain
        elif mode == 'unlearn':
            dataset = self.dataset_unlearn
        
        assert dataset.is_scale == False, "dataset should be unscaled"
        self.X = np.concatenate((dataset.feature.numpy(), np.ones((dataset.feature.shape[0], 1))), axis = 1)
        self.Y = dataset.target.numpy()
    
    def fit_regressor(self, mode):
        self.select_dataset(mode)
        A = np.matmul(self.X.T, self.X)
        b = np.matmul(self.X.T, self.Y)
        parameter = np.linalg.solve(A, b) # solve for a matrix
        
        return parameter.T
        
    def cal_grad(self, mode, parameter):
        # calculate the gradint of the loss function
        # gradient of (y-theta * x)^T(y-theta*x): -2(y-theta*x)*x^T
        # or in batch form: -2 * (Y*T - theta * X^T) * X
        self.select_dataset(mode)
        
        grad = -2 * (self.Y.T - parameter @ self.X.T) @ self.X / self.X.shape[0] / self.Y.shape[1] # ! for the mse loss, the output is also averaged
        
        # in the neural network, the weight matrix is first flattened, then the bias is added to the end
        grad_bias = grad[:, -1]
        grad_weight = grad[:, :-1]
        grad = np.concatenate((grad_weight.flatten(), grad_bias))
        
        return grad
    
    def cal_hessian(self, mode, parameter):
        # the hessian can be written as ((14 * 89), (14 * 89), ...,  (14 * 89)) the total number is 14*89
        # e.g. it is a tensor of (14 * 89, 14, 89)
        self.select_dataset(mode)
        hessian_list = [] # the hessian is a list of 14 * 89 matrices, each matrix is the hessian with respect to one parameter in the weight matrix
        J = np.zeros_like(parameter)
        
        # hessian on weight
        for m in range(parameter.shape[0]):
            for n in range(parameter.shape[1] - 1):
                J[m,n] = 1
                hessian = 2 * J @ self.X.T @ self.X
                # first flatten the weight matrix then add the bias
                hessian_weight = hessian[:, :-1]
                hessian_bias = hessian[:, -1]
                hessian = np.concatenate((hessian_weight.flatten(), hessian_bias))
                hessian_list.append(hessian / self.X.shape[0] / self.Y.shape[1])
                J[m,n] = 0 # reset
        
        # hessian on bias
        for m in range(parameter.shape[0]):
            J[m,-1] = 1
            hessian = 2 * J @ self.X.T @ self.X
            hessian_weight = hessian[:, :-1]
            hessian_bias = hessian[:, -1]
            hessian = np.concatenate((hessian_weight.flatten(), hessian_bias))
            hessian_list.append(hessian / self.X.shape[0] / self.Y.shape[1])
            J[m,-1] = 0 # reset
        
        return np.array(hessian_list)
    
    
class LinearModel(nn.Module):
    
    def __init__(self, regressor):
        super().__init__()
        
        weights = regressor.coef_
        bias = regressor.intercept_
        self.fc = nn.Linear(weights.shape[1], weights.shape[0], bias=True)
        self.fc.weight = nn.Parameter(torch.tensor(weights))
        self.fc.bias = nn.Parameter(torch.tensor(bias))
    
    def forward(self, x):
        return self.fc(x)

def evaluate(model, dataset):
    feature = dataset.feature.numpy()
    target = dataset.target.numpy()
    if isinstance(model, nn.Module):
        model.eval()
        with torch.no_grad():
            pred = model(torch.tensor(feature)).numpy()
    elif isinstance(model, LinearRegression):
        pred = model.predict(feature)
    else:
        raise ValueError("model should be either nn.Module or LinearRegression")
    
    mape = np.mean(np.abs((target - pred) / target)) * 100
    
    return mape 
    
def return_module(model, method, hessian_loader, grad_loader, device = 'cpu', damp = 0.0):
    """
    method: direct, cg, lissa
    damp = 0.0 as the linear model is guaranteed to be strong convex
    n the hessian vector product:
        hessian loader: the loader for the dataset to calculate the hessian
        grad loader: the loader for the dataset to calculate the gradient
    """
    
    class MyObjective(BaseObjective):

        def train_outputs(self, model, batch):
            # batch is a tuple of (feature, target)
            return model(batch[0])

        def train_loss_on_outputs(self, outputs, batch):
            return F.mse_loss(outputs, batch[1])  # mean reduction required

        def train_regularization(self, params):
            # no regularization
            return 0. * torch.square(params.norm())

        # training loss by default taken to be 
        # train_loss_on_outputs + train_regularization

        def test_loss(self, model, params, batch):
            return F.mse_loss(model(batch[0]), batch[1])
    
    if method == 'direct':
    
        module = AutogradInfluenceModule(
            model=model,
            objective=MyObjective(),  
                train_loader=hessian_loader, # for exact unlearning, we need to calculate the hessian on the remain dataset
                test_loader=grad_loader,  # this can be replaced by unlearn_loader which is also exact
                device=device,
                damp=damp,
                check_eigvals = False
            )
    
    elif method == 'cg':
        module = CGInfluenceModule(
            model=model,
            objective=MyObjective(),  
                train_loader=hessian_loader, # for exact unlearning, we need to calculate the hessian on the remain dataset
                test_loader=grad_loader,  # this can be replaced by unlearn_loader which is also exact
                device=device,
                damp=damp
            )
    
    return module
    

if __name__ == "__main__":
    
    from dataset import return_dataset
    from torch.utils.data import DataLoader
    import argparse
    import json
    
    with open("config.json") as f:
        config = json.load(f)
        
    batch_size = config['batch_size']
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-u', '--unlearn_prop', type=float)
    args = parser.parse_args()
    
    # dataset
    unlearn_prop = args.unlearn_prop
    dataset_train = return_dataset("case14", False, True, 'train')
    dataset_test = return_dataset("case14", False, True, 'test')
    dataset_remain = return_dataset("case14", False, True, 'remain', unlearn_prop)
    dataset_unlearn = return_dataset("case14", False, True, 'unlearn', unlearn_prop)
    
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)
    loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)
    loader_remain = DataLoader(dataset_remain, batch_size = batch_size, shuffle = False)
    loader_unlearn = DataLoader(dataset_unlearn, batch_size = batch_size, shuffle = False)

    print('length of datasets: train {} test {} unlearn {} remain {}'.format(len(dataset_train), len(dataset_test), len(dataset_unlearn), len(dataset_remain)))
    
    sklearn_solver = SklearnSolver(dataset_train, dataset_test, dataset_remain, dataset_unlearn)
    analytic_solver = AnalyticSolver(dataset_train, dataset_test, dataset_remain, dataset_unlearn)
    
    """
    train set
    """
    regressor = sklearn_solver.fit_regressor(mode = 'train')
    parameter_analytic = analytic_solver.fit_regressor(mode = 'train')
    weight_analytic = parameter_analytic[:, :-1]
    bias_analytic = parameter_analytic[:, -1]
    
    # print(regressor.coef_.shape, regressor.intercept_.shape)
    # print(weight_analytic.shape, bias_analytic.shape)
    print('train on train set')
    print('max difference between weights: ', np.max(np.abs(regressor.coef_ - weight_analytic)))
    print('max difference between bias: ', np.max(np.abs(regressor.intercept_ - bias_analytic)))
    print('weight shape: {}, bias shape: {}'.format(regressor.coef_.shape, regressor.intercept_.shape))
    linear_model = LinearModel(regressor)
    
    # print the shape of model parameter
    for name, param in linear_model.named_parameters():
        print(name, param.shape)
    
    mape_train = evaluate(regressor, dataset_train)
    mape_model = evaluate(linear_model, dataset_train)
    
    assert np.isclose(mape_model, mape_train), "mape should be the same"
    
    mape_test = evaluate(regressor, dataset_test)
    print('mape on train set: ', mape_train, '%')
    print('mape on test set: ', mape_test, '%')    
    
    print("=====================================")
    
    """
    test the gradient of module and analytic solver
    """
    grad_analytic = analytic_solver.cal_grad(mode = 'unlearn', parameter = parameter_analytic)
    # ! the gradient calculated from module is concatenated as (88, 88, ... 88, 14), 
    # the ihvp is the same
    # the total number of parameters is 88 * 14 + 14 = 1246
    module_direct = return_module(linear_model, 'direct', loader_unlearn, loader_unlearn) 
    grad_direct = module_direct.test_loss_grad(test_idxs = range(len(dataset_unlearn)))
    
    abs_diff = np.abs(grad_analytic - grad_direct.numpy()) / np.abs(grad_analytic)
    print('shape of the gradients: ', grad_analytic.shape, grad_direct.shape)
    print('max difference between gradient: ', np.max(abs_diff) * 100, '%')
    
    print("=====================================")
    """
    test the hessian of module and analytic solver
    """
    hessian_analytic = analytic_solver.cal_hessian(mode = 'unlearn', parameter = parameter_analytic)
    hessian_direct = module_direct.hessian
    
    abs_diff = np.abs(hessian_analytic - hessian_direct.numpy())
    print('shape of the hessian: ', hessian_analytic.shape, hessian_direct.shape)
    print('max difference between hessian: ', np.max(abs_diff))
    
    print("=====================================")
    
    """
    unlearning
        hessian: remain
        gradient: remain
    """
    
    print("Unlearning performance. Hessian: remain, grad: remain.")
    
    # directly train on the remain set
    regressor_remain = sklearn_solver.fit_regressor(mode = 'remain')
    mape_remain_remain = evaluate(regressor_remain, dataset_remain)
    mape_unlearn_remain = evaluate(regressor_remain, dataset_unlearn)
    mape_test_remain = evaluate(regressor_remain, dataset_test)
    
    mape_remain_train = evaluate(regressor, dataset_remain)
    mape_unlearn_train = evaluate(regressor, dataset_unlearn)
    mape_test_train = evaluate(regressor, dataset_test)
    
    print("mape on remain set: train vs remain", mape_remain_train, mape_remain_remain, "%")
    print("mape on test set: train vs remain ", mape_test_train, mape_test_remain, "%")
    print("mape on unlearn set: train vs remain ", mape_unlearn_train, mape_unlearn_remain, "%")
    
    parameter_remain = np.concatenate((regressor_remain.coef_.flatten(), regressor_remain.intercept_))
    parameter_train = np.concatenate((regressor.coef_.flatten(), regressor.intercept_))
    
    # print("Unlearning on the hessian: remain and gradient: remain")
    # # unlearn using the analytic solver
    # grad_remain = analytic_solver.cal_grad(mode = 'remain', parameter = parameter_analytic)
    # hessian_remain = analytic_solver.cal_hessian(mode = 'remain', parameter = parameter_analytic)
    # ihvp_remain = np.linalg.inv(hessian_remain) @ grad_remain
    # parameter_remain_ = parameter_train - ihvp_remain
    # # print('max difference between parameters: ', np.max(np.abs(parameter_remain - parameter_remain_)))
    # assert np.isclose(parameter_remain, parameter_remain_, atol=1e-4).all(), "parameter should be the same"
    # print("The analytic unlearning is correct")
    
    # # unlearn using the module direct
    # module_direct = return_module(linear_model, 'direct', loader_remain, loader_remain)
    # grad_remain = module_direct.test_loss_grad(test_idxs = range(len(dataset_remain)))
    # ihvp_remain = module_direct.inverse_hvp(grad_remain)
    # parameter_remain_ = parameter_train - ihvp_remain.numpy()
    # # print('max difference between parameters: ', np.max(np.abs(parameter_remain - parameter_remain_)))
    # assert np.isclose(parameter_remain, parameter_remain_, atol=1e-4).all(), "parameter should be the same"
    # print("The direct unlearning is correct")
    
    # # unlearn using cg
    # module_cg = return_module(linear_model, 'cg', loader_remain, loader_remain)
    # grad_remain = module_cg.test_loss_grad(test_idxs = range(len(dataset_remain)))
    # ihvp_remain = module_cg.inverse_hvp(grad_remain)
    # parameter_remain_ = parameter_train - ihvp_remain.numpy()
    # # print('max difference between parameters: ', np.max(np.abs(parameter_remain - parameter_remain_)))
    # assert np.isclose(parameter_remain, parameter_remain_, atol=1e-4).all(), "parameter should be the same"
    # print("The CG unlearning is correct")
    
    # # using stest function
    # model_cg = return_module(linear_model, 'cg', loader_remain, loader_remain)
    # ihvp_remain = model_cg.stest(test_idxs = range(len(dataset_remain)))
    # parameter_remain_ = parameter_train - ihvp_remain.numpy()
    # assert np.isclose(parameter_remain, parameter_remain_, atol=1e-4).all(), "parameter should be the same"
    # print(' The stest is correct')
    
    # print("=====================================")
    
    """
    unlearning
        hessian: remain
        gradient: unlearn
    """
    print("Unlearning performance. Hessian: remain, grad: unlearn.")
    model_cg = return_module(linear_model, 'cg', hessian_loader = loader_remain, grad_loader = loader_unlearn)
    ihvp = model_cg.stest(test_idxs = range(len(dataset_unlearn))) * len(dataset_unlearn) / len(dataset_remain)
    parameter_remain_ = parameter_train + ihvp.numpy() # ! note here should be plus
    assert np.isclose(parameter_remain, parameter_remain_, atol=1e-4).all(), "parameter should be the same"
    
    print("=====================================")
    
    # """
    # unlearning
    #     hessian: train
    #     gradient: remain
    # """
    # print("Unlearning performance. Hessian: train, grad: remain.")
    # model_cg = return_module(linear_model, 'cg', hessian_loader = loader_train, grad_loader = loader_remain)
    # # find the hessian on the train dataloader and gradient on the test dataloader
    # # ! all averaged based on their size, the same to the paper setting
    # ihvp_remain = model_cg.stest(test_idxs = range(len(dataset_remain)))
    # ihvp_remain_noscale = ihvp_remain * len(dataset_remain) / len(dataset_train)
    # parameter_remain = parameter_train - ihvp_remain.numpy()
    # parameter_remain_noscale = parameter_train - ihvp_remain_noscale.numpy()
    
    # print('diff without rescale: ', np.linalg.norm(parameter_remain_ - parameter_remain, ord = 2))
    # print('diff with rescale: ', np.linalg.norm(parameter_remain_noscale - parameter_remain, ord = 2))
    