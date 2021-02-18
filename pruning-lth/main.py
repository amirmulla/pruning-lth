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
    prune_ratio = args.prune_ratio / 100
    if args.prune_ratio_conv is not None:
        prune_ratio_conv = args.prune_ratio_conv / 100
    else:
        prune_ratio_conv = None

    batch_size = args.batch_size
    prune_output_layer = bool(args.prune_output_layer)
    winning_ticket_reinit = bool(args.winning_ticket_reinit)
    prune_init = args.prune_init
    rounds = args.prune_rounds
    dp_ratio = args.dp_ratio / 100

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
        model = LeNet(dp_ratio=dp_ratio)
        optim_type = 'adam'
        lr = 1.2e-3
    elif model_type == 'conv4':
        model = Conv_4(dp_ratio=dp_ratio)
        optim_type = 'adam'
        lr = 3e-4
    elif model_type == 'vgg19':
        model = VGG(dp_ratio=dp_ratio)
        optim_type = 'sgd'
        lr = 0.01

    print(model)

    #######################################
    # Identifying winning tickets.        #
    #######################################

    p_conv = None

    print(
        f'Prune {model_type} model, with {approach} pruning and {prune_init} init,using {optim_type} optimizer with learning rate={lr} and with pruning rate of {100 * prune_ratio:.1f}%')

    if approach == "iterative":

        # Step 1
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
                        f'model-{model_type}_batchsz-{batch_size}_dp_ratio-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-{100 * (1 - sparsity):.1f}',
                        epochs=epochs, lr=lr, optimizer_type=optim_type,
                        train_loader=train_loader, test_loader=test_loader,
                        model_dir=model_dir, results_dir=results_dir)

            if round < (rounds - 1):
                # Step 3
                p = pow(prune_ratio, (1 / (round + 1)))
                if prune_ratio_conv is not None:
                    p_conv = pow(prune_ratio_conv, (1 / (round + 1)))
                print(f'pruning rate: {100 * p:.1f}%')
                prune_model(model=model, prune_ratio=p, prune_ratio_conv=p_conv, prune_method=method,
                            prune_output_layer=prune_output_layer)

                # Step 4
                if prune_init == "rewind":
                    rewind_model_weights(model, init_weights)
                elif prune_init == "random":
                    model.rand_initialize_weights()

        print(f'Model pruning, took {time.time() - t0: .2f} seconds')

    elif approach == "oneshot":
        # Step 0
        t0 = time.time()
        init_prune_model(model)
        sparsity_l = []
        for round in range(rounds):
            sparsity = calc_model_sparsity(model) / 100
            if sparsity != 0:
                sparsity_l.append(sparsity)
            p = pow(prune_ratio, (1 / (round + 1)))
            if prune_ratio_conv is not None:
                p_conv = pow(prune_ratio_conv, (1 / (round + 1)))
            prune_model(model=model, prune_ratio=p, prune_ratio_conv=p_conv, prune_method=method,
                        prune_output_layer=prune_output_layer)

        # Step 1
        init_prune_model(model)
        print_sparsity(model)
        train_model(model,
                    f'model-{model_type}_batchsz-{batch_size}_dp_ratio-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-100',
                    epochs=epochs, lr=lr, optimizer_type=optim_type,
                    train_loader=train_loader, test_loader=test_loader,
                    model_dir=model_dir, results_dir=results_dir)

        for sparsity in sparsity_l:
            # Step 2
            prune_model(model=model, prune_ratio=float(sparsity), prune_ratio_conv=prune_ratio_conv, prune_method=method,
                        prune_output_layer=prune_output_layer)

            print_sparsity(model)
            train_model(model,
                        f'model-{model_type}_batchsz-{batch_size}_dp_ratio-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-{100 * (1 - sparsity):.1f}',
                        epochs=epochs, lr=lr, optimizer_type=optim_type,
                        train_loader=train_loader, test_loader=test_loader,
                        model_dir=model_dir, results_dir=results_dir)

            init_prune_model(model)
            model_name = model_dir + '/' + f'model-{model_type}_batchsz-{batch_size}_dp_ratio-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-100' + '.pt'
            model.load_state_dict(torch.load(model_name))

        print(f'Model pruning, took {time.time() - t0: .2f} seconds')

    elif approach == "random":
        # Step 0
        t0 = time.time()
        init_prune_model(model)
        sparsity_l = []
        for round in range(rounds):
            sparsity = calc_model_sparsity(model) / 100
            if sparsity != 0:
                sparsity_l.append(sparsity)
            p = pow(prune_ratio, (1 / (round + 1)))
            if prune_ratio_conv is not None:
                p_conv = pow(prune_ratio_conv, (1 / (round + 1)))
            prune_model(model=model, prune_ratio=p, prune_ratio_conv=p_conv, prune_method=method,
                        prune_output_layer=prune_output_layer)

        # Step 1
        init_prune_model(model)

        for sparsity in sparsity_l:
            # Step 2
            prune_model(model=model, prune_ratio=float(sparsity), prune_ratio_conv=prune_ratio_conv, prune_method=method,
                        prune_output_layer=prune_output_layer)
            print_sparsity(model)
            train_model(model,
                        f'model-{model_type}_batchsz-{batch_size}_dp_ratio-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-{100 * (1 - sparsity):.1f}',
                        epochs=epochs, lr=lr, optimizer_type=optim_type,
                        train_loader=train_loader, test_loader=test_loader,
                        model_dir=model_dir, results_dir=results_dir)

            init_prune_model(model)
            model.rand_initialize_weights()

        print(f'Model pruning, took {time.time() - t0: .2f} seconds')

    #############################################
    # Random initialization of winning tickets. #
    #############################################

    if prune_init == "rewind" and winning_ticket_reinit:
        print(
            f'Train and evaluate random re-initialization of winning/random tickets of {model_type} model:')

        i = 0
        for sparsity in sparsity_l:
            print('Random training round: {}'.format(i + 1))
            t0 = time.time()
            # load winning ticket
            model_name = model_dir + '/' + f'model-{model_type}_batchsz-{batch_size}_dp_ratio-{dp_ratio}_approach-{approach}_method-{method}_init-{prune_init}_remainweights-{100 * (1 - sparsity):.1f}' + '.pt'
            model.load_state_dict(torch.load(model_name))
            model.rand_initialize_weights()
            print_sparsity(model)
            train_model(model,
                        f'model-{model_type}_batchsz-{batch_size}_dp_ratio-{dp_ratio}_approach-{approach}_method-{method}_init-randomreinit_remainweights-{100 * (1 - sparsity):.1f}',
                        epochs=epochs, lr=lr, optimizer_type=optim_type,
                        train_loader=train_loader, test_loader=test_loader,
                        model_dir=model_dir, results_dir=results_dir)
            i += 1

        print(f'Model Training with random init, took {time.time() - t0: .2f} seconds')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", default="lenet", type=str, help="lenet | conv4 | vgg19")
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--prune_approach", default="iterative", type=str, help="iterative | oneshot | random")
    parser.add_argument("--prune_method", default="local", type=str, help="local | global")
    parser.add_argument("--train_epochs", default=64, type=int)
    parser.add_argument("--prune_ratio", default=20, type=int, help="Initial pruning ratio (0-100)")
    parser.add_argument("--prune_output_layer", default=1, type=int, help="Apply pruning to output layer")
    parser.add_argument("--prune_rounds", default=10, type=int, help="Number of pruning rounds")
    parser.add_argument("--prune_init", default="rewind", type=str, help="rewind | random")
    parser.add_argument("--winning_ticket_reinit", default=0, type=int, help="Random reinitialization of winning "
                                                                             "tickets (from pruning with rewind)")
    parser.add_argument("--dp_ratio", default=20, type=int, help="Dropout ratio (0-100)")
    parser.add_argument("--prune_ratio_conv", default=None, type=int, help="Initial pruning ratio (0-100) for "
                                                                           "Convolution layers")

    args = parser.parse_args()

    main(args)
