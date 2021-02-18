import pickle
import time

import numpy as np
import torch
from torch import nn
from torch import optim


###########################
# Model Evaluation        #
###########################

def eval_model(model, criterion, loader):
    running_loss = 0.0
    class_correct = list(0. for i in range(10))
    class_total = list(0. for i in range(10))

    for data, target in loader:
        data, target = data.cuda(), target.cuda()
        output = model(data)
        loss = criterion(output, target)
        running_loss += loss.item() * data.size(0)
        _, pred = torch.max(output, 1)
        correct = np.squeeze(pred.eq(target.data.view_as(pred)))
        for i in range(len(target)):
            label = target.data[i]
            class_correct[label] += correct[i].item()
            class_total[label] += 1

    out_loss = running_loss / len(loader.sampler)
    out_acc = np.sum(class_correct) / np.sum(class_total)

    return out_loss, out_acc


###########################
# Model Training          #
###########################

def train_model(model, model_name, epochs=20, optimizer_type='adam', lr=0.001, results_dir=None, model_dir=None,
                train_loader=None, test_loader=None):
    model_res_path = results_dir + '/' + model_name + '.pkl'
    model_name = model_dir + '/' + model_name + '.pt'

    model.cuda()

    # Define the loss and optimizer
    criterion = nn.CrossEntropyLoss().cuda()
    if optimizer_type == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=lr)
    elif optimizer_type == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)

    # init results data structure
    results = {'accs_losses': {'train_loss': [], 'train_acc': [],
                               'test_loss': [], 'test_acc': []},
               'min_valid_loss_data': {'min_valid_loss_val': np.Inf, 'min_valid_loss_iter': 0}}

    test_loss_min = np.Inf

    for epoch in range(epochs):
        t0 = time.time()

        # Train the model
        model.train()
        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()
            optimizer.zero_grad()
            output = model(images)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

        # Evaluate the model
        model.eval()
        with torch.no_grad():
            train_loss, train_acc = eval_model(model, criterion, train_loader)
            test_loss, test_acc = eval_model(model, criterion, test_loader)

        print(
            f'Epoch: {epoch + 1} ({time.time() - t0:.2f} seconds) Train Loss: {train_loss:.6f} Train Accuracy: {train_acc:.6f} Test Loss: {test_loss:.6f} Test Accuracy: {test_acc:.6f}')

        # save model if validation loss has decreased
        if test_loss <= test_loss_min:
            print(f'Test loss decreased ({test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')
            torch.save(model.state_dict(), model_name)
            test_loss_min = test_loss
            # save to results
            results['min_valid_loss_data']['min_valid_loss_val'] = test_loss_min
            results['min_valid_loss_data']['min_valid_loss_iter'] = epoch + 1

        results['accs_losses']['train_loss'].append(train_loss)
        results['accs_losses']['train_acc'].append(train_acc)
        results['accs_losses']['test_loss'].append(test_loss)
        results['accs_losses']['test_acc'].append(test_acc)

    print('Saving results...')
    with open(model_res_path, 'wb') as handle:
        pickle.dump(results, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print('Saved')