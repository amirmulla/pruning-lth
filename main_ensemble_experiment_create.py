import time

import torch
from functional_models.architectures import *
from functional_models.eval_train import *
from functional_models.pruning_funcs import *
from torch import optim
from torchvision import datasets, transforms

results_dir = 'results'
model_dir = 'saved_models'

##############################
# Parameters.                #
##############################

model_type = 'conv4'
approach = 'random'
method = 'local'
batch_size = 64
prune_ratio = 0.2
prune_ratio_conv = 20

if prune_ratio_conv is not None:
    prune_ratio_conv = prune_ratio_conv / 100
else:
    prune_ratio_conv = None

batch_size = 64
prune_output_layer = 1
winning_ticket_reinit = 0
prune_init = 'rewind'
rounds = 6
dp_ratio = 0

find_matching_tickets = 0
stabilize_epochs = 0
use_lr_scheduler = 0

# Experiments
epochs_init = 30
pruned_epochs = 30
experiment_nums = range(10)
boost_prune_round = 5

##############################
# Train and Test Loaders.    #
##############################

num_workers = 2

if model_type == 'lenet':
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

    train_data = datasets.MNIST('./dataset/', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = datasets.MNIST('./dataset/', train=False, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

elif model_type == 'conv4' or model_type == 'vgg19':
    transform_train = transforms.Compose(
        [transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])
    transform_test = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)), ])

    train_data = datasets.CIFAR10('./dataset/', train=True, download=True, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)

    test_data = datasets.CIFAR10('./dataset/', train=False, download=True, transform=transform_test)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)

##############################
# Init Network.              #
##############################

# initialize the NN and define optimizer/learning rate
if model_type == 'lenet':
    model = LeNet(dp_ratio=dp_ratio)
    optim_type = 'adam'
    lr = 1.2e-3
    weight_decay = 0
elif model_type == 'conv4':
    model = Conv_4(dp_ratio=dp_ratio)
    optim_type = 'adam'
    lr = 3e-4
    weight_decay = 0
elif model_type == 'vgg19':
    model = VGG(dp_ratio=dp_ratio)
    optim_type = 'sgd'
    lr = 0.01
    weight_decay = 1e-4

print(model)

#########################################
# Identifying winning/matching tickets. #
#########################################

print(f'Prune {model_type} model, with {approach} pruning and {prune_init} init,using {optim_type} optimizer with learning rate={lr} and with pruning rate of {100 * prune_ratio:.1f}%')

# Experiment 
for experi in experiment_nums:
    print(f'Expriment number {experi}:')    
    t0 = time.time()
    if approach == 'oneshot':
        model_name_100 = f'model-{model_type}_batchsz-{batch_size}_dp-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-100.0_experiment-{experi}'
        init_weights_name_100 = f'model-{model_type}_batchsz-{batch_size}_dp-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-100.0_experiment-{experi}_init-weights'
        if os.path.exists(os.path.join(results_dir, model_name_100+'.pkl')):
            print(f"Unpruned model already exists. Loading...")
            init_prune_model(model)
            model.load_state_dict(torch.load(os.path.join(model_dir, model_name_100+'.pt')))
            init_weights = torch.load(os.path.join(model_dir, init_weights_name_100+'.pt'))
            print(f"Loaded model and initial weights.")
        else:
        # Step 1
            init_prune_model(model)
            init_weights = save_model_weights(model)
            torch.save(init_weights, os.path.join(model_dir, init_weights_name_100+'.pt'))  # save initial weights

            sparsity = calc_model_sparsity(model) / 100
            print_sparsity(model)

            print('First round, initial training:')
            train_model(model,
                        model_name_100,
                        epochs=epochs_init, lr=lr, weight_decay=weight_decay, optimizer_type=optim_type,
                        use_lr_scheduler=use_lr_scheduler,
                        train_loader=train_loader, test_loader=test_loader,
                        model_dir=model_dir, results_dir=results_dir)
            init_prune_model(model)
    elif approach == 'random':
        init_prune_model(model)
        model.rand_initialize_weights()
    
    # Step 2
    for round in range(boost_prune_round, rounds):
        for i in range(round):
            p = pow(prune_ratio, (1 / (i + 1)))
            if prune_ratio_conv is not None:
                p_conv = pow(prune_ratio_conv, (1 / (i + 1)))
            prune_model(model=model, prune_ratio=p, prune_ratio_conv=p_conv, prune_method=method,
                        prune_output_layer=prune_output_layer)

        sparsity = calc_model_sparsity(model) / 100
        print_sparsity(model)

        model_name = f'model-{model_type}_batchsz-{batch_size}_dp-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-{100 * (1 - sparsity):.1f}_experiment-{experi}'

        if os.path.exists(os.path.join(results_dir, model_name+'.pkl')):
            print(f"Model already exists for round {round}. Skipping")
        else:
            if approach == 'oneshot':
                # Step 3
                rewind_model_weights(model, init_weights)

            Step 4
            train_model(model,
                        model_name,
                        epochs=pruned_epochs, lr=lr, weight_decay=weight_decay, optimizer_type=optim_type,
                        use_lr_scheduler=use_lr_scheduler,
                        train_loader=train_loader, test_loader=test_loader,
                        model_dir=model_dir, results_dir=results_dir)
        # Step 5
        init_prune_model(model)
        if approach == 'oneshot':
            model.load_state_dict(torch.load(os.path.join(model_dir, model_name_100 + '.pt')))
        elif approach == 'random':
            model.rand_initialize_weights()      

    print(f'Model pruning, took {time.time() - t0: .2f} seconds')