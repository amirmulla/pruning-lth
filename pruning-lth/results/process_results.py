import matplotlib.pyplot as plt
import numpy as np
import pickle
import os


def load_file(fn):
    with open(fn, 'rb') as f:
        result_dict = pickle.load(f)
    return result_dict


def parse_filename(fn):
    fn_metadata = {}
    fn_no_ext, ext = os.path.splitext(fn)
    metadata_parts = fn_no_ext.rsplit('_')
    for md_part in metadata_parts:
        md_pair = md_part.rsplit('-')
        if md_pair[0].find('batchsz') != -1 or md_pair[0].find('remainweights') != -1:
            md_pair[1] = float(md_pair[1])
        fn_metadata[md_pair[0]] = md_pair[1]
    return fn_metadata


def load_model_data(model_type, approach_type, init_type):
    # data to keep
    out_data = {'weights_remaining': [], 'train_acc': [], 'test_acc': [],
                'min_valid_loss_iter': [], 'min_valid_loss_val': [],
                'min_valid_loss_train_acc': [], 'min_valid_loss_test_acc': []}
    # filenames
    files_in_path = os.listdir(os.getcwd())
    # print(files_in_path)
    relevant_files = []
    relevant_files_metadata = []
    # keep filenames with model name in them
    for fn in files_in_path:
        fn_no_ext, ext = os.path.splitext(fn)
        if all([fn.find(f'model-{model_type}_') != -1, fn.find(f'approach-{approach_type}_') != -1,
                fn.find(f'init-{init_type}_') != -1]):
            relevant_files.append(fn)
    if len(relevant_files) == 0:
        return None, out_data
    metadata_dict = parse_filename(relevant_files[-1])  # keep metadata in variable
    metadata_dict.pop('remainweights')  # keep only common metadata
    # load data and append to data structure
    for fn in relevant_files:
        # parse filename
        fn_no_ext, _ = os.path.splitext(fn)
        metadata_parts = fn_no_ext.rsplit('_')
        prune_part = metadata_parts[-1].rsplit('-')
        # store metadata
        out_data['weights_remaining'].append(float(prune_part[1]))
        # load file
        res_dict = load_file(fn)
        out_data['train_acc'].append(res_dict['accs_losses']['train_acc'])
        out_data['test_acc'].append(res_dict['accs_losses']['test_acc'])
        out_data['min_valid_loss_val'].append(res_dict['min_valid_loss_data']['min_valid_loss_val'])
        min_valid_loss_iter = res_dict['min_valid_loss_data']['min_valid_loss_iter']
        out_data['min_valid_loss_iter'].append(min_valid_loss_iter)
        out_data['min_valid_loss_train_acc'].append(out_data['train_acc'][-1][min_valid_loss_iter - 1])
        out_data['min_valid_loss_test_acc'].append(out_data['test_acc'][-1][min_valid_loss_iter - 1])
    # sort data according to weights_remaining
    sorted_indices = np.flip(np.argsort(out_data['weights_remaining']))
    for key in out_data:
        out_data[key] = [out_data[key][i] for i in sorted_indices]
    return metadata_dict, out_data


def plot_from_files(model_type, approach_type_list, init_type_list, x_field, y_field, agg_plots=False, wp_to_plot=None,
                    iter_num=None, ylims=None):
    x = None
    plt.figure()
    for approach_type in approach_type_list:
        for init_type in init_type_list:
            metadata, plot_data = load_model_data(model_type=model_type.lower(), approach_type=approach_type,
                                                  init_type=init_type)
            if metadata is None:
                continue
            # get data to plot
            if x_field == 'weights_remaining':
                x_ticks = plot_data[x_field]
                x = [ii for ii in range(len(x_ticks))]
                if iter_num is None:
                    y = plot_data[y_field]
                else:  # here we need to get data form y_field across different lists
                    y = []
                    lists = plot_data[y_field]
                    for l in lists:
                        y.append(l[iter_num])
                if init_type != 'random':
                    plt.plot(x, y, label=f'{approach_type}')
                else:
                    plt.plot(x, y, label=f'{approach_type} randomly initialized')
                plt.xticks(x, x_ticks)
            elif x_field == 'iterations' and (
                    y_field == 'test_acc' or y_field == 'train_acc'):  # take plots as a function of the iterations
                # for different % of remaining weights
                if agg_plots:
                    all_plots = []
                    all_rem_wts = []
                    for rem_wts, y in zip(plot_data['weights_remaining'], plot_data[y_field]):
                        if wp_to_plot is not None and rem_wts not in wp_to_plot:
                            continue
                        all_plots.append(y)
                        all_rem_wts.append(rem_wts)

                        x = np.arange(len(y))
                        for f_y, f_rem_wts in zip(all_plots, all_rem_wts):
                            plt.plot(x, f_y, label=f'{f_rem_wts:.1f}')
                        plt.legend(loc='best')
                        plt.grid()
                        plt.suptitle(f'Results for {model_type}')
                        plt.xlabel('# iterations')
                        if y_field == 'test_acc':
                            plt.ylabel('Test Acc')
                        if y_field == 'train_acc':
                            plt.ylabel('Train Acc')
                        plt.ylim(ylims)

                        plt.figure()
                else:
                    plot_num_in_fig = 0
                    for rem_wts, y in zip(plot_data['weights_remaining'], plot_data[y_field]):
                        if rem_wts == 100:
                            y_100 = y
                        if iter_num is not None:
                            y = y[:iter_num]
                        if ylims is not None and max(y) < ylims[0]:
                            continue
                            # x = metadata['batchsz']*np.arange(len(y))
                        x = np.arange(len(y))
                        plt.plot(x, y, label=f'{rem_wts:.1f}')
                        plot_num_in_fig += 1
                        if plot_num_in_fig == 5:
                            plt.legend(loc='best')
                            plt.grid()
                            plt.suptitle(f'Results for {model_type}')
                            plt.xlabel('# iterations')
                            if y_field == 'test_acc':
                                plt.ylabel('Test Acc')
                            if y_field == 'train_acc':
                                plt.ylabel('Train Acc')
                            plt.ylim(ylims)
                            plt.figure()
                            plot_num_in_fig = 0
                            plt.plot(x, y_100, label=f'{100:.1f}')
    plt.legend(loc='best')
    plt.grid()
    plt.suptitle(f'Results for {model_type}')
    plt.xlabel('# iterations')
    if y_field == 'test_acc':
        plt.ylabel('Test Acc')
    if y_field == 'train_acc':
        plt.ylabel('Train Acc')
    plt.ylim(ylims)
    if x is not None:
        plt.xlim((x[0], x[-1]))
    if iter_num is not None:
        if iter_num == -1:
            abs_iter_num = len(l)
        else:
            abs_iter_num = (iter_num + 1)
        return abs_iter_num


# Commented out IPython magic to ensure Python compatibility.
# LeNet Main plot type
model_type = 'LeNet'
method_type = 'local'  # used for lenet

# Test Acc as a function of iterations for different remaining weights %
ylims = (0.97, 1)
# ylim_bottom = None
plot_from_files(model_type, approach_type_list=['iterative'], init_type_list=['rewind'], x_field='iterations',
                y_field='test_acc', ylims=ylims)

# Early stop iteration as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['iterative', 'oneshot', 'random'], init_type_list=['rewind'],
                x_field='weights_remaining', y_field='min_valid_loss_iter')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Early Stop Iteration')

# Test Acc at early stop as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['iterative', 'oneshot', 'random'], init_type_list=['rewind', 'random'],
                x_field='weights_remaining', y_field='min_valid_loss_test_acc')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Test Acc. at Early Stop Iteration')
plt.ylim((0.97, 1))

# Train Acc at early stop as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['iterative', 'oneshot', 'random'], init_type_list=['rewind', 'random'],
                x_field='weights_remaining', y_field='min_valid_loss_train_acc')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Train Acc. at Early Stop Iteration')
plt.ylim((0.97, 1))

# Test Acc at the end of the training a function of remaining weights %
abs_iter_num = plot_from_files(model_type, approach_type_list=['iterative', 'oneshot'],
                               init_type_list=['rewind', 'random'], x_field='weights_remaining', y_field='test_acc',
                               iter_num=-1)
plt.xlabel('% of Remaining Weights')
plt.ylabel(f'Test Acc. at Final Iteration ({int(abs_iter_num)})')
plt.ylim((0.97, 1))

# NOW ONLY FOR ONESHOT
# Test Acc at early stop as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['oneshot'], x_field='weights_remaining',
                init_type_list=['rewind', 'random'], y_field='min_valid_loss_test_acc')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Test Acc. at Early Stop Iteration')
plt.ylim((0.97, 1))

# Train Acc at early stop as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['oneshot'], x_field='weights_remaining',
                init_type_list=['rewind', 'random'], y_field='min_valid_loss_train_acc')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Train Acc. at Early Stop Iteration')
plt.ylim((0.97, 1))

# Commented out IPython magic to ensure Python compatibility.
# Conv4 Main plot type
model_type = 'Conv4'
method_type = 'local'  # used for lenet

# Test Acc as a function of iterations for different remaining weights %
ylims = (0.75, 0.85)
# ylim_bottom = None
plot_from_files(model_type, approach_type_list=['iterative'], init_type_list=['rewind', 'random'], x_field='iterations',
                y_field='test_acc', ylims=ylims)

# Early stop iteration as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['iterative', 'oneshot', 'random'], init_type_list=['rewind', 'random'],
                x_field='weights_remaining', y_field='min_valid_loss_iter')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Early Stop Iteration')

# Test Acc at early stop as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['iterative', 'oneshot', 'random'], init_type_list=['rewind', 'random'],
                x_field='weights_remaining', y_field='min_valid_loss_test_acc')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Test Acc. at Early Stop Iteration')
plt.ylim((0.75, 0.85))

# Train Acc at early stop as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['iterative', 'oneshot', 'random'], init_type_list=['rewind', 'random'],
                x_field='weights_remaining', y_field='min_valid_loss_train_acc')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Train Acc. at Early Stop Iteration')
# plt.ylim((0.75,0.85))

# Test Acc at the end of the training a function of remaining weights %
abs_iter_num = plot_from_files(model_type, approach_type_list=['iterative', 'oneshot'],
                               init_type_list=['rewind', 'random'], x_field='weights_remaining', y_field='test_acc',
                               iter_num=-1)
plt.xlabel('% of Remaining Weights')
plt.ylabel(f'Test Acc. at Final Iteration ({int(abs_iter_num)})')
plt.ylim((0.75, 0.85))

# NOW ONLY FOR ONESHOT
# Test Acc at early stop as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['oneshot'], x_field='weights_remaining',
                init_type_list=['rewind', 'random'], y_field='min_valid_loss_test_acc')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Test Acc. at Early Stop Iteration')
plt.ylim((0.75, 0.85))

# Train Acc at early stop as a function of remaining weights %
plot_from_files(model_type, approach_type_list=['oneshot'], x_field='weights_remaining',
                init_type_list=['rewind', 'random'], y_field='min_valid_loss_train_acc')
plt.xlabel('% of Remaining Weights')
plt.ylabel('Train Acc. at Early Stop Iteration')
plt.ylim((0.75, 0.85))
