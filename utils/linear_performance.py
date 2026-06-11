"""
evaluate the influence of each train sample on the performance of the model
"""
from sklearn.linear_model import LinearRegression
from torch.utils.data import Dataset

def fit_regressor(dataset):
    feature = dataset.feature.numpy()
    target = dataset.target.numpy()
    reg = LinearRegression().fit(feature, target)
    return reg

class NewDataset(Dataset):
    
    def __init__(self, feature, target):
        self.feature = feature
        self.target = target
    
    def __len__(self):
        return len(self.target)
    
    def __getitem__(self, idx):
        return self.feature[idx], self.target[idx]

if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from torch_influence import BaseObjective
    from torch_influence import AutogradInfluenceModule, CGInfluenceModule, LiSSAInfluenceModule
    import torch
    import torch.nn.functional as F
    import json
    import argparse
    from torch.utils.data import DataLoader, TensorDataset
    from dataset import return_dataset
    from linear_reg import SklearnSolver, LinearModel
    from linear_reg import evaluate
    
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
    # dataset_remain = return_dataset("case14", False, True, 'remain', unlearn_prop)
    # dataset_unlearn = return_dataset("case14", False, True, 'unlearn', unlearn_prop)
    
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)
    loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)
    # loader_remain = DataLoader(dataset_remain, batch_size = batch_size, shuffle = False)
    # loader_unlearn = DataLoader(dataset_unlearn, batch_size = batch_size, shuffle = False)

    print('length of datasets: train {} test {}'.format(len(dataset_train), len(dataset_test)))
    
    class MyObjective(BaseObjective):

        def train_outputs(self, model, batch):
            # batch is a tuple of (feature, target)
            return model(batch[0])

        def train_loss_on_outputs(self, outputs, batch):
            return F.mse_loss(outputs, batch[1])  # mean reduction required, averaged over the sample dimension as well

        def train_regularization(self, params):
            # no regularization
            return 0. * torch.square(params.norm())

        # training loss by default taken to be 
        # train_loss_on_outputs + train_regularization

        def test_loss(self, model, params, batch):
            # mape loss for the test set
            return torch.mean(torch.abs(model(batch[0]) - batch[1]) / batch[1])
    
    regressor = fit_regressor(dataset_train)
    linear_model = LinearModel(regressor)
    
    module = CGInfluenceModule(
            model=linear_model,
            objective=MyObjective(),  
                train_loader=loader_train, # for exact unlearning, we need to calculate the hessian on the remain dataset
                test_loader=loader_test,  # this can be replaced by unlearn_loader which is also exact
                device='cpu',
                damp=0.
            )
    """
    construct the unlearn dataset  which is most helpful to the test loss
    """
    # influence on the test set
    # The most helpful point is that which, if removed, most increases the loss 
    # Conversely, the most harmful test point is that which most decreases the test loss if removed.
    influences = module.influences(train_idxs = range(len(dataset_train)), test_idxs = range(len(dataset_test)))
    
    unlearn_no = int(args.unlearn_prop * len(dataset_train))
    
    helpful_index = torch.argsort(influences, descending = True)[:unlearn_no].numpy()
    # harmful_index = torch.argsort(influences, descending = False)[:unlearn_no]
    
    remain_index = [i for i in range(len(dataset_train)) if i not in helpful_index]
    
    # remain and helpful should be disjoint
    assert len(set(remain_index).intersection(set(helpful_index))) == 0
    
    dataset_unlearn = NewDataset(dataset_train.feature[helpful_index], dataset_train.target[helpful_index])
    dataset_remain = NewDataset(dataset_train.feature[remain_index], dataset_train.target[remain_index])
    
    regressor_remain = fit_regressor(dataset_remain)
    mape_test_train = evaluate(regressor, dataset_test)
    mape_test_remain = evaluate(regressor_remain, dataset_test)
    
    print('mape test train: {:.4f} mape test remain: {:.4f}'.format(mape_test_train, mape_test_remain))
    