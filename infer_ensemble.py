import torch
from torch import nn
import torch.nn.utils.prune as prune
import numpy as np
from torch import optim
import copy
import time
import os

from functional_models.architectures import *
from functional_models.eval_train import *
from functional_models.pruning_funcs import *

from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler

results_dir = 'results'
ensemble_model_dir = 'saved_models'


# Commented out IPython magic to ensure Python compatibility.
####################################
# Import functions and Models      #
####################################


#############################
# Ensemble Model Evaluation #
#############################

def eval_model_ens(model_list, criterion, loader, ens_type = 'avg'):
    ens_size = len(model_list)
    running_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in loader:
        pred_list = []
        data, target = data.cuda(), target.cuda()
        for model in model_list:
            model.cuda()
            model.eval()
            output = model(data)
            if ens_type == 'maj':
                _, pred = torch.max(output, 1)
            elif ens_type == 'avg':
                pred = output/ens_size
            pred_list.append(pred)
        # make decision by taking all classifiers into account
        if ens_size > 1:
            if ens_type == 'maj':
                pred_list_stack = torch.stack(pred_list, dim=0).detach().cpu().numpy()
                pred_size = pred_list_stack.shape
                total_pred = []
                for ii in range(pred_size[1]):
                    total_pred.append(np.argmax(np.bincount(pred_list_stack[:,ii])))
                running_loss = np.nan
            elif ens_type == 'avg':
                total_output = torch.stack(pred_list, dim=0).sum(dim=0)
                _, total_pred = torch.max(total_output, 1)
                loss = criterion(total_output, target)
                running_loss += loss.item() * data.size(0)     
        else:
            if ens_type == 'maj':
                running_loss = np.nan
            elif ens_type == 'avg':
                total_output = pred_list[0]
                _, total_pred = torch.max(total_output, 1)
   
        correct = np.squeeze(total_pred.eq(target.data.view_as(total_pred)))
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    out_loss = running_loss / len(loader.sampler)
    out_acc = np.sum(class_correct) / np.sum(class_total)

    return out_loss, out_acc

#############################
# Ensemble Model Inferecnce #
#############################

def infer_ensemble_model(model_list, model_name, results_dir=None, test_loader=None, ens_type='avg'):
    model_res_path = results_dir + '/' + model_name + '.pkl'

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()

    # init results data structure
    results = {'accs_losses': {'train_loss': [], 'train_acc': [],
                               'test_loss': [], 'test_acc': []},
               'min_valid_loss_data': {'min_valid_loss_val': np.Inf, 'min_valid_loss_iter': 0}}

    t0 = time.time()

    # Evaluate the model
    with torch.no_grad():
        test_loss, test_acc = eval_model_ens(model_list, criterion, test_loader, ens_type=ens_type)

    print(
        f'({time.time() - t0:.2f} seconds) Test Loss: {test_loss:.6f} Test Accuracy: {test_acc:.6f}')

    # save to results
    results['min_valid_loss_data']['min_valid_loss_val'] = test_loss
    results['min_valid_loss_data']['min_valid_loss_iter'] = 1


    results['accs_losses']['train_loss'].append(np.nan)
    results['accs_losses']['train_acc'].append(np.nan)
    results['accs_losses']['test_loss'].append(test_loss)
    results['accs_losses']['test_acc'].append(test_acc)

    print('Saving results...')
    with open(model_res_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved')

##############################
# Train and Test Loaders.    #
##############################

batch_size = 64
model_type = 'conv4'
num_workers = 2

if model_type == 'lenet':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    test_data = datasets.MNIST('./dataset/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

elif model_type == 'conv4' or model_type == 'vgg19':
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    test_data = datasets.CIFAR10('./dataset/', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

################################
# Load All Models and Organize #
################################

# filenames
files_in_path = os.listdir(ensemble_model_dir)

models_fns = []
models_list = {}
models_experiment_100 = [] # not part of the ensemble
# keep filenames with model name in them
for fn in files_in_path:      
    fn_no_ext, ext = os.path.splitext(fn)          
    if all([fn.find(f'experiment-') != -1]):
        models_fns.append(fn)
models_fns.sort()
for fn in models_fns:        
    model_metadata = parse_filename(fn)
    if model_metadata['experiment'] == 100:
        models_experiment_100.append(fn)
    else:
        if model_metadata['remainweights'] not in models_list.keys():
            models_list[model_metadata['remainweights']] = []
        models_list[model_metadata['remainweights']].append(Conv_4(dp_ratio=0))
        init_prune_model(models_list[model_metadata['remainweights']][-1])
        models_list[model_metadata['remainweights']][-1].load_state_dict(torch.load(os.path.join(ensemble_model_dir, fn)))
        print(f"Loaded {fn}")


##############################
# Create Ensembles and Infer #
##############################

for key in models_list.keys():
    maximal_bb_num = int(100/key) + 1
    print(maximal_bb_num)
    for bb_mult in range(1,maximal_bb_num):
        if bb_mult > len(models_list[key]):
            continue
        ensemble_list = models_list[key][:bb_mult]
        total_rel_size = key*bb_mult
        print(f"Using an ensemble of {bb_mult} building blocks of {key}%, summing up to a {total_rel_size:.2f}% model. Sort of.")
        model_name = f"model-{model_type}_buildingblock-{key}_totalsize-{total_rel_size:.2f}"
        infer_ensemble_model(ensemble_list, model_name, results_dir=results_dir, test_loader=test_loader, ens_type='avg')