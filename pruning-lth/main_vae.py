import time

import matplotlib.pyplot as plt
import numpy as np
import torch
from functional_models.architectures import *
from functional_models.eval_train import *
from functional_models.pruning_funcs import *
from torch import optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms

results_dir = 'results'
model_dir = 'saved_models'

###########################
# Datasets                #
###########################

batch_size = 64

# Define a transform
transform = transforms.Compose([transforms.ToTensor(), ])
train_data = datasets.MNIST('./dataset/', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=2)
test_data = datasets.MNIST('./dataset/', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=2)


def criterion(y, x, mu, log_var):
    BCE = torch.nn.functional.binary_cross_entropy(y, x, reduction='sum')
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - torch.exp(log_var))
    return BCE + KLD


def EvaluateModel(model, loader):
    running_loss = 0.0
    for input, target in loader:
        input, _ = input.cuda(), target.cuda()
        input = input.view(input.size(0), -1)
        output, mu, log_var = model(input)
        loss = criterion(output, input, mu, log_var)
        running_loss += loss.item()

    running_loss /= len(loader.dataset)
    return running_loss


def train_and_evaluate_model(model, model_name, epochs=20, lr=0.001):
    model_name = model_dir + '/' + model_name + '.pt'

    # Create
    model.cuda()

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        t0 = time.time()

        # Train the model
        model.train()
        for input, target in train_loader:
            input, _ = input.cuda(), target.cuda()
            input = input.view(input.size(0), -1)
            model.zero_grad()
            optimizer.zero_grad()
            output, mu, log_var = model(input)
            loss = criterion(output, input, mu, log_var)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            train_loss = EvaluateModel(model, train_loader)

        model_result.append(train_loss)

        print('Epoch: {} ({:.2f} seconds) Train Loss: {:.6f}'.format(epoch + 1, time.time() - t0, train_loss))

    print('Saving VAE...')
    torch.save(model.state_dict(), model_name)
    print('VAE Saved')


###############################
# Pruning VAE Lottery Ticket. #
###############################

batch_size = 64
epochs = 30
lr = 0.001
rounds = 10
prune_ratio = 0.2

print('Find Lottery Tickets:')
t0 = time.time()
model = MNIST_VAE()

# Step 1
init_prune_model(model)
init_weights = save_model_weights(model)

models_results = []

sparsity_l = []
for round in range(rounds):
    print('Prune Round: {}'.format(round + 1))
    t0 = time.time()
    sparsity = calc_model_sparsity(model) / 100
    sparsity_l.append(sparsity)
    print_sparsity(model)

    model_result = []

    # Step 2
    train_and_evaluate_model(model, f'MNIST_VAE_Remainweights-{100 * (1 - sparsity):.1f}', epochs=epochs, lr=lr)

    models_results.append(model_result)

    model_name = model_dir + '/' + f'MNIST_VAE_Remainweights-{100 * (1 - sparsity):.1f}' + '.pt'
    model.load_state_dict(torch.load(model_name))

    if round < (rounds - 1):
        # Step 3
        p = pow(prune_ratio, (1 / (round + 1)))
        print(f'pruning rate: {100 * p:.1f}%')
        prune_model(model=model, prune_ratio=p)

        # Step 4
        rewind_model_weights(model, init_weights)

print(f'Model pruning, took {time.time() - t0: .2f} seconds')

###############################
# Plot VAE    Lottery Ticket. #
###############################

x = []
y = []
for sparsity, model_result in zip(sparsity_l, models_results):
    remaining = float(sparsity * 100)
    x.append(remaining)
    y.append(min(model_result))

plt.figure()
plt.plot(x, y)
plt.grid()
plt.xlabel('Sparsity [%]')
plt.ylabel('Train Loss')
plt.legend()

######################################
# Random Pruning VAE Lottery Ticket. #
######################################

batch_size = 64
epochs = 30
lr = 0.001
rounds = 10
prune_ratio = 0.2

print('Find Lottery Tickets:')
t0 = time.time()
model = MNIST_VAE()

# Step 1
init_prune_model(model)
init_weights = save_model_weights(model)

random_models_results = []

random_sparsity_l = []

t0 = time.time()
for round in range(rounds):
    print('Random Prune Round: {}'.format(round + 1))

    for i in range(round):
        p = pow(prune_ratio, (1 / (i + 1)))
        prune_model(model=model, prune_ratio=p)

    sparsity = calc_model_sparsity(model) / 100
    random_sparsity_l.append(sparsity)
    print_sparsity(model)

    model_result = []
    train_and_evaluate_model(model, f'MNIST_VAE_RANDOM_Remainweights-{100 * (1 - sparsity):.1f}', epochs=epochs, lr=lr)
    random_models_results.append(model_result)

    init_prune_model(model)
    model.rand_initialize_weights()

print(f'Model pruning, took {time.time() - t0: .2f} seconds')

###############################
# Plot VAE  Random Pruning.   #
###############################

x_rand = []
y_rand = []
for sparsity, model_result in zip(random_sparsity_l, random_models_results):
    remaining = float(sparsity * 100)
    x_rand.append(remaining)
    y_rand.append(min(model_result))

plt.figure()
plt.plot(x, y, label='Winning Tickets')
plt.plot(x_rand, y_rand, label='Random Pruning')
plt.grid()
plt.xlabel('VAE Sparsity [%]')
plt.ylabel('Reconstruction Loss')
plt.ylim(100, 120)
plt.legend()

# Commented out IPython magic to ensure Python compatibility.
##############################
# Plot VAE Reconstruction    #
##############################

model = MNIST_VAE()
init_prune_model(model)
model.cuda()

# Obtain one batch of test images
test_image, test_label = next(iter(test_loader))
test_image = test_image.cuda()
test_label = test_label.cuda()

for sparsity in sparsity_l:
    model_name = model_dir + '/' + f'MNIST_VAE_Remainweights-{100 * (1 - sparsity):.1f}' + '.pt'
    model.load_state_dict(torch.load(model_name))
    print_sparsity(model)
    test_output, _, _ = model(torch.flatten(test_image, start_dim=1))

    x = test_image.cpu()
    y = test_output.cpu()

    x = x.numpy()
    y = y.view(-1, 1, 28, 28)
    y = y.detach().numpy()

    # Plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for x, row in zip([x, y], axes):
        for img, ax in zip(x, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

    model_name = model_dir + '/' + f'MNIST_VAE_RANDOM_Remainweights-{100 * (1 - sparsity):.1f}' + '.pt'
    model.load_state_dict(torch.load(model_name))
    print_sparsity(model)
    test_output, _, _ = model(torch.flatten(test_image, start_dim=1))

    x = test_image.cpu()
    y = test_output.cpu()

    x = x.numpy()
    y = y.view(-1, 1, 28, 28)
    y = y.detach().numpy()

    # Plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=2, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    for x, row in zip([x, y], axes):
        for img, ax in zip(x, row):
            ax.imshow(np.squeeze(img), cmap='gray')
            ax.get_xaxis().set_visible(False)
            ax.get_yaxis().set_visible(False)

#####################################
# Generate Random Images using VAE  #
#####################################

batch_size = 20

sample = torch.randn(batch_size, 50)
sample = sample.cuda()

for sparsity in sparsity_l:
    model_name = model_dir + '/' + f'MNIST_VAE_Remainweights-{100 * (1 - sparsity):.1f}' + '.pt'
    model.load_state_dict(torch.load(model_name))
    print_sparsity(model)

    with torch.no_grad():
        output = model.decode(sample)
        y = output.cpu()

    y = y.view(batch_size, 1, 28, 28)
    y = y.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    y_iter = iter(y)

    for ax in axes:
        img = next(y_iter)
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    model_name = model_dir + '/' + f'MNIST_VAE_RANDOM_Remainweights-{100 * (1 - sparsity):.1f}' + '.pt'
    model.load_state_dict(torch.load(model_name))
    print_sparsity(model)

    with torch.no_grad():
        output = model.decode(sample)
        y = output.cpu()

    y = y.view(batch_size, 1, 28, 28)
    y = y.detach().numpy()

    # plot the first ten input images and then reconstructed images
    fig, axes = plt.subplots(nrows=1, ncols=10, sharex=True, sharey=True, figsize=(25, 4))

    # input images on top row, reconstructions on bottom
    y_iter = iter(y)

    for ax in axes:
        img = next(y_iter)
        ax.imshow(np.squeeze(img), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
