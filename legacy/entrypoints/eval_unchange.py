"""
evaluate the performance unchanged unlearning on the following models
1. linear model
2. nn_conv
3. nn_mlpmixer

based on the criteria
1. mape unchange
2. cost unchange
"""

from torch.utils.data import DataLoader
from utils import return_dataset, DatasetWithWeight, evaluate, reconstruct_model, flatten_model
from func_operation import return_unlearn_datasets, return_trained_model, return_module, return_core_datasets, return_dataset_for_nn_affine
import numpy as np
import torch
import time
import cvxpy as cp
import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")
def main(cfg: DictConfig):

    unlearn_prop = cfg.unlearn_prop
    batch_size = cfg.data.batch_size_eval
    model_type = cfg.model.type
    influence_dir = cfg.model.influence_dir
    train_loss = cfg.model.train_loss
    l1_constraints = cfg.model.l1_constraints
    linf_constraint = 1.
    criteria = cfg.criteria
    
    print(f"model type: {model_type}, unlearn prop: {unlearn_prop}")

    # dataset
    influences = torch.from_numpy(np.load(influence_dir)).float()
    dataset_train, dataset_test = return_dataset(cfg)
    if 'nn' in model_type:
        dataset_core, dataset_sensitive = return_core_datasets(cfg, dataset_to_be_split=dataset_train)
        dataset_train, dataset_test = return_dataset_for_nn_affine(cfg, dataset_sensitive, dataset_test)
    
    dataset_unlearn, dataset_remain, unlearn_index, remain_index = return_unlearn_datasets(
                                            influences=influences, 
                                            unlearn_prop=unlearn_prop, dataset_to_be_unlearn=dataset_train, 
                                            mode=cfg.unlearn_mode,
                                            config=cfg)
    
    loader_train = DataLoader(dataset_train, batch_size = batch_size, shuffle = False)
    loader_test = DataLoader(dataset_test, batch_size = batch_size, shuffle = False)
    loader_remain = DataLoader(dataset_remain, batch_size = batch_size, shuffle = False)  # only unlearn using the remain set formulation

    print('Load is scaled:\n', dataset_train.is_scale, dataset_test.is_scale, dataset_unlearn.is_scale, dataset_remain.is_scale)
    print('train dataset shape: ', dataset_train.feature.shape, 'test dataset shape: ', dataset_test.feature.shape)

    dataset_collection = {'remain': dataset_remain, 'unlearn': dataset_unlearn, 'test': dataset_test}
    
    """
    original model
    """
    model_ori = return_trained_model(cfg,
                                    model_type = model_type + "-affine" if 'nn' in model_type else model_type,
                                    dataset_train = dataset_train, is_spo = False)
    model_ori.eval()
    parameter_ori = flatten_model(model_ori)

    print('getting the performance of the original model')
    mse_ori = evaluate(model_ori, dataset_collection, loss = 'mse')
    mape_ori = evaluate(model_ori, dataset_collection, loss = 'mape')
    cost_ori = evaluate(model_ori, dataset_collection, loss = 'cost', case_config = cfg.case, with_mispatch=True) # also record the prediction mismatch

    cost_ori_ = {"remain": cost_ori["remain"], "unlearn": cost_ori["unlearn"], "test": cost_ori["test"]}
    print('mse ori: \n', mse_ori)
    print('mape ori: \n', mape_ori)
    print('cost ori: \n', cost_ori_)

    """
    unlearning driven by mse
    """
    module_unlearn = return_module(
        cfg,
        loss_type_dict = {"train":train_loss, "test": train_loss},
        loader_dict={"train": loader_remain, "test": loader_remain},
        model=model_ori, method = 'cg', watch_progress=False
    )

    ihvp = module_unlearn.stest(test_idxs=range(len(dataset_remain))).numpy()
    parameter_unlearn_ = parameter_ori - ihvp
    model_unlearn = reconstruct_model(model = model_ori, flattened_params = parameter_unlearn_)

    print('getting the performance of the unlearned model')
    mse_unlearn = evaluate(model = model_unlearn,
                    dataset_collection= dataset_collection, loss = "mse")
    mape_unlearn = evaluate(model = model_unlearn,
                    dataset_collection= dataset_collection, loss = "mape")
    cost_unlearn = evaluate(model = model_unlearn,
                    dataset_collection= dataset_collection, loss = "cost", with_mispatch=True,
                    case_config=cfg.case)

    cost_unlearn_ = {"remain": cost_unlearn["remain"], "unlearn": cost_unlearn["unlearn"], "test": cost_unlearn["test"]}

    print('mse unlearn: \n', mse_unlearn)
    print('mape unlearn: \n', mape_unlearn)
    print('cost unlearn: \n', cost_unlearn_)

    """
    find the influence score of each train sample to the test sample
    """
    print('getting the influence score of each train sample to the test sample')
    # construct module on train = remain (to calculate the inverse hessian), test = train (to calculate gradient)
    module_train = return_module(cfg,
                            loss_type_dict={"train": train_loss, "test": train_loss}, 
                            loader_dict={"train": loader_remain, "test": loader_train},
                            model=model_ori, method = 'cg')
    
    # calcualte the gradient on the train dataset
    start_train = time.time()
    grad_train_all = []
    for i in range(len(dataset_train)):
        grad_train_all.append(module_train.test_loss_grad(test_idxs=[i]).numpy())
    print("time for calculating train grad: ", round(time.time() - start_train, 2))

    # generate average gradient on the test set using the performance criteria
    if criteria != "cost":
        model_test = return_trained_model(cfg,
                                        model_type = model_type + "-affine" if 'nn' in model_type else model_type, 
                                        dataset_train = dataset_train, is_spo = False)
    else:
        model_test = return_trained_model(cfg,
                                        model_type = model_type + "-affine" if 'nn' in model_type else model_type, 
                                        dataset_train = dataset_train, is_spo = True)

    module_test = return_module(cfg,
                                loss_type_dict={"train": "mse", "test": criteria}, # train here is not used
                                loader_dict={"train": loader_train, "test": loader_test},
                                model=model_test, method = 'cg')

    start_time = time.time()
    grad_test_ave = module_test.test_loss_grad(test_idxs=range(len(dataset_test)))
    print("time for calculating test grad: ", round(time.time() - start_time,2))

    # calculate M matrix defined in the paper. hessian: remain and train loss, grad: test and test loss
    # ! we do not use spo model which is too slow
    start_time = time.time()
    M = -module_train.inverse_hvp(vec = grad_test_ave).numpy()
    print("time for calculating M: ", round(time.time() - start_time, 2))

    # calculate the score: the inlfuence of each train sample to the expected performance of the test dataset
    scores = []
    for grad in grad_train_all:
        scores.append(grad @ M)

    scores = np.array(scores) / len(dataset_train) # average over the size of train dataset
    scores_remain = scores[remain_index]
    scores_unlearn = scores[unlearn_index]
    print('performance change of unlearning (positive for helpful): {}'.format(-np.sum(scores_unlearn)))

    scores_summary = {
        "remain": scores_remain,
        "unlearn": scores_unlearn,
        "mismatch_remain": cost_ori["remain_gen_mismatch"],
        "mismatch_unlearn": cost_ori["unlearn_gen_mismatch"] 
    }

    np.save(f'{cfg.simulation_dir}/{model_type}/unchange_{unlearn_prop}_{criteria}_scores.npy', scores_summary, allow_pickle=True)


    """
    PAMU and TAMU
    """

    log = {
        "mse_test_ori": mse_ori["test"],
        "mse_unlearn_ori": mse_ori["unlearn"],
        "mse_remain_ori": mse_ori["remain"],
        "mape_test_ori": mape_ori["test"],
        "mape_unlearn_ori": mape_ori["unlearn"],
        "mape_remain_ori": mape_ori["remain"],
        "cost_test_ori": cost_ori["test"],
        "cost_unlearn_ori": cost_ori["unlearn"],
        "cost_remain_ori": cost_ori["remain"],
        "mse_test_unlearn": mse_unlearn["test"],
        "mse_unlearn_unlearn": mse_unlearn["unlearn"],
        "mse_remain_unlearn": mse_unlearn["remain"],
        "mape_test_unlearn": mape_unlearn["test"],
        "mape_unlearn_unlearn": mape_unlearn["unlearn"],
        "mape_remain_unlearn": mape_unlearn["remain"],
        "cost_test_unlearn": cost_unlearn["test"],
        "cost_unlearn_unlearn": cost_unlearn["unlearn"],
        "cost_remain_unlearn": cost_unlearn["remain"],
        "l1_constraints": l1_constraints,
        "linf_constraint": [linf_constraint] * len(l1_constraints) if isinstance(linf_constraint, float) else linf_constraint,
        "mse_test": [],
        "mse_unlearn": [],
        "mse_remain": [],
        "mape_test": [],
        "mape_unlearn": [],
        "mape_remain": [],
        "cost_test": [],
        "cost_unlearn": [],
        "cost_remain": [],
        "parameter_diff": []
    }

    print('performance unchanged unlearning...')
    for constraint in l1_constraints:
    
        print("========================================================================")
        print('constraint:', constraint)

        # re-weight the remain dataset
        eps_remain = cp.Variable(len(dataset_remain))
        obj = cp.Minimize(cp.scalar_product(eps_remain, scores_remain)) # we do not need to use abs here
        # average so that can be generalized to different dataset size
        cons = [cp.norm(eps_remain - 1, 1) <= constraint * len(dataset_remain), cp.norm(eps_remain - 1, np.inf) <= linf_constraint]
        prob = cp.Problem(obj, cons)
        '''修改这里'''
        '''prob.solve(verbose=False, solver=cp.GUROBI)'''
        prob.solve(verbose=False, solver=cp.ECOS, max_iters=2000)
        print('status:', prob.status)
        eps_remain = eps_remain.value
        eps_all = np.zeros(len(dataset_train))
        eps_all[remain_index] = eps_remain
        eps_all[unlearn_index] = -1  # the unlearn dataset is with weight -1
        ihvp = 0
        
        # we need to contruct the module so that the weighted mse loss can be applied
        dataset_remain_with_weight = DatasetWithWeight(dataset_remain, eps_remain)
        loader_remain_with_weight = DataLoader(dataset_remain_with_weight, batch_size = batch_size, shuffle = False)
        module_unlearn = return_module(cfg,
                            loss_type_dict={"train": train_loss, "test": train_loss},  # using the training loss to unlearn
                            loader_dict={"train": loader_remain_with_weight, "test": loader_remain_with_weight},
                            model=model_ori, method = 'cg', with_weight = True)
        
        ihvp = module_unlearn.stest(test_idxs=range(len(dataset_remain)))
        ihvp = ihvp.numpy()
        parameter_unlearn = parameter_ori - ihvp
        
        # print(ihvp)
        model_unlearn = reconstruct_model(model_ori, parameter_unlearn)
        mse_unlearn = evaluate(model_unlearn, dataset_collection, loss = 'mse')
        mape_unlearn = evaluate(model_unlearn, dataset_collection, loss = 'mape')
        cost_unlearn = evaluate(model_unlearn, dataset_collection, loss = 'cost', case_config=cfg.case)
        
        print(np.linalg.norm(eps_remain - 1, 1), np.linalg.norm(eps_remain - 1, np.inf))
            
        print('mse unlearn: \n', mse_unlearn)
        print('mape unlearn: \n', mape_unlearn)
        print('cost unlearn: \n', cost_unlearn)
        parameter_diff = np.linalg.norm(parameter_unlearn_ - parameter_unlearn, 2)
        print('parameter difference: ', parameter_diff)
        
        log['mse_test'].append(mse_unlearn['test'])
        log['mse_unlearn'].append(mse_unlearn['unlearn'])
        log['mse_remain'].append(mse_unlearn['remain'])
        
        log['mape_test'].append(mape_unlearn['test'])
        log['mape_unlearn'].append(mape_unlearn['unlearn'])
        log['mape_remain'].append(mape_unlearn['remain'])
        
        log['cost_test'].append(cost_unlearn['test'])
        log['cost_unlearn'].append(cost_unlearn['unlearn'])
        log['cost_remain'].append(cost_unlearn['remain'])
        
        log['parameter_diff'].append(parameter_diff)

    np.save(f'{cfg.simulation_dir}/{model_type}/unchange_{unlearn_prop}_{criteria}.npy', log)    


if __name__ == "__main__":
    main()