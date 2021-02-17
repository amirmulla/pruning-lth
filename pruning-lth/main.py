import argparse
import time

import torch
from torchvision import datasets, transforms
from functional_models.eval_train import *
from functional_models.architectures import *
from functional_models.pruning_funcs import *


def main(args):
    results_dir = 'results'
    model_dir = 'saved_models'

    ##############################
    # Parameters.                #
    ##############################

    model_type = args.model_type
    approach = args.prune_approach
    method = args.prune_method
    epochs = args.train_epochs
    prune_amount = args.prune_ratio
    batch_size = args.batch_size
    prune_output_layer = args.prune_output_layer

    if approach == 'oneshot':
        rounds = 2
    else:
        rounds = args.iter_prune_rounds

    ##############################
    # Train and Test Loaders.    #
    ##############################

    if model_type == 'lenet':
        transform = transforms.ToTensor()
        train_data = datasets.MNIST('./dataset/', train=True, download=True, transform=transform)
        test_data = datasets.MNIST('./dataset/', train=False, download=True, transform=transform)
    elif model_type == 'conv4' or model_type == 'vgg19':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
        train_data = datasets.CIFAR10('./dataset/', train=True, download=True, transform=transform)
        test_data = datasets.CIFAR10('./dataset/', train=False, download=True, transform=transform)

    # prepare data loaders
    num_workers = 0
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    ##############################
    # Init Network.              #
    ##############################

    # initialize the NN and define optimizer/learning rate
    if model_type == 'lenet':
        model = LeNet()
        optim_type = 'adam'
        lr = 1.2e-3
    elif model_type == 'conv4':
        model = Conv_4()
        optim_type = 'adam'
        lr = 3e-4
    elif model_type == 'vgg19':
        model = VGG()
        optim_type = 'sgd'
        lr = 0.1

    print(model)
    #######################################
    # Identifying winning tickets.        #
    #######################################

    init = 'rewind'

    print(
        f'Prune {model_type} model, with {approach} pruning and {init} init,using {optim_type} optimizer with learning rate={lr} and with pruning rate of {100 * prune_amount:.1f}%')

    # Step 1
    init_model_weights(model, model_type)
    init_prune_model(model)
    init_weights = save_model_weights(model)

    sparsity_l = []
    for round in range(rounds):
        print('Prune Round: {}'.format(round + 1))
        t0 = time.time()
        sparsity = calc_model_sparsity(model) / 100
        sparsity_l.append(sparsity)
        print_sparsity(model)

        # Step 2
        train_model(model,
                    f'model-{model_type}_batchsz-{batch_size}_approach-{approach}_method-{method}_init-{init}_remainweights-{100 * (1 - sparsity):.1f}',
                    epochs=epochs, lr=lr,
                    train_loader=train_loader, test_loader=test_loader,
                    model_dir=model_dir, results_dir=results_dir)

        if round < (rounds - 1):
            # Step 3
            p = pow(prune_amount, (1 / (round + 1)))
            print(f'pruning rate: {100 * p:.1f}%')
            prune_model(model, p, method, prune_output_layer)

            # Step 4
            rewind_model_weights(model, init_weights)

    print(f'Model pruning, took {time.time() - t0: .2f} seconds')

    #############################################
    # Random initialization of winning tickets. #
    #############################################

    init = 'random'

    print(
        f'Train and evaluate winning/random ticket {model_type} model, with {approach} pruning and {init} init, using {optim_type} optimizer with learning rate={lr} and with pruning rate of {100 * prune_amount:.1f}%')

    i = 0
    for round in range(rounds):
        print('Random training round: {}'.format(round + 1))
        t0 = time.time()
        # load winning ticket
        sparsity = sparsity_l[i]
        model_name = model_dir + '/' + f'model-{model_type}_batchsz-{batch_size}_approach-{approach}_method-{method}_init-rewind_remainweights-{100 * (1 - sparsity):.1f}' + '.pt'
        model.load_state_dict(torch.load(model_name))
        init_model_weights(model, model_type)
        print_sparsity(model)
        train_model(model,
                    f'model-{model_type}_batchsz-{batch_size}_approach-{approach}_method-{method}_init-{init}_remainweights-{100 * (1 - sparsity):.1f}',
                    epochs=epochs, lr=lr,
                    train_loader=train_loader, test_loader=test_loader,
                    model_dir=model_dir, results_dir=results_dir)
        i += 1

    print(f'Model Training with random init, took {time.time() - t0: .2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type",default="lenet", type=str, help="lenet | conv4 | vgg19")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--prune_approach", default="iterative", type=str, help="iterative | oneshot")
    parser.add_argument("--prune_method", default="local", type=str, help="local | global")
    parser.add_argument("--train_epochs", default=64, type=int)
    parser.add_argument("--prune_ratio", default=0.1, type=int, help="Initial pruning ratio (0-1)")
    parser.add_argument("--prune_output_layer", default=True, type=bool, help="Apply pruning to output layer")
    parser.add_argument("--iter_prune_rounds", default=10, type=int, help="# of rounds in iterative pruning")
    args = parser.parse_args()

    main(args)
