import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import torch
import json
import datetime
import models

#=======================================================================================
def txt2list(filename):
    lines_list = []
    with open(filename, 'r') as txt:
        for line in txt:
            lines_list.append(line.rstrip('\n'))
    return lines_list


def key(dictio,val):
    for c,v in dictio.items():
        if v == val:
            return(c)


def calibration_bins_size(nb_bins_size, gaps, min_value):
    half = int(nb_bins_size/2)
    bins_size1, bins_size2 = [0], []
    inf = min_value
    for i in range(len(gaps)): 
        sup = inf + int(gaps[i]*half/len(gaps))
        bins_size1 += list(range(inf, sup, gaps[i]))
        inf = sup
    shift = bins_size1[-1] + gaps[-1]
    assert bins_size1[-1] > 1500, "Error: top limit bins has to reach packet's maximum size (1500)"
  
    for i in range(len(gaps)):
        sup = inf + int(gaps[i]*half/len(gaps))
        bins_size1 += list(range(inf, sup, gaps[i]))
        inf = sup
    return bins_size1+bins_size2, shift


def collate_fn(data):
    X_batch = torch.tensor(np.array([d[0] for d in data]))
    y_batch = torch.tensor([d[1] for d in data])
    
    return X_batch, y_batch
#=======================================================================================

def current2firing_time(x, tau=20, thr=0.2, tmax=1.0, epsilon=1e-7):
    """ Computes first firing time latency for a current input x assuming the charge time of a current based LIF neuron.

    Args:
    x -- The "current" values

    Keyword args:
    tau -- The membrane time constant of the LIF neuron to be charged
    thr -- The firing threshold value 
    tmax -- The maximum time returned 
    epsilon -- A generic (small) epsilon > 0

    Returns:
    Time to first spike for each "current" x
    """
    idx = x<thr
    x = np.clip(x,thr+epsilon,1e9)
    T = tau*np.log(x/(x-thr))       # firing rate = 1/(T+tau_ref), see LIF slides of Vosoughi
    T[idx] = tmax
    return T


def data_generator(X, y, batch_size, device, dtype, shuffle=True ):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.
    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y,dtype=np.int)
    number_of_batches = len(X)//batch_size
    sample_index = np.arange(len(X))

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter<number_of_batches:
        batch_index = sample_index[batch_size*counter:batch_size*(counter+1)]

        X_batch = torch.tensor(X[batch_index],device=device, dtype=dtype)
        y_batch = torch.tensor(labels_[batch_index],device=device)

        yield X_batch, y_batch

        counter += 1


def sparse_data_generator(X, y, batch_size, nb_steps, nb_units, time_step, device, dtype, shuffle=True):
    """ This generator takes datasets in analog format and generates spiking network input as sparse tensors.

    Args:
        X: The data ( sample x event x 2 ) the last dim holds (time,neuron) tuples
        y: The labels
    """

    labels_ = np.array(y, dtype=np.int)
    number_of_batches = len(X) // batch_size
    sample_index = np.arange(len(X))

    # compute discrete firing times
    tau_eff = 20e-3 / time_step
    firing_times = np.array(current2firing_time(X, tau=tau_eff, tmax=nb_steps), dtype=np.int)
    unit_numbers = np.arange(nb_units)

    if shuffle:
        np.random.shuffle(sample_index)

    total_batch_count = 0
    counter = 0
    while counter < number_of_batches:
        batch_index = sample_index[batch_size * counter:batch_size * (counter + 1)]

        coo = [[] for i in range(3)]
        for bc, idx in enumerate(batch_index):
            c = firing_times[idx] < nb_steps
            times, units = firing_times[idx][c], unit_numbers[c]

            batch = [bc for _ in range(len(times))]
            coo[0].extend(batch)
            coo[1].extend(times)
            coo[2].extend(units)

        i = torch.LongTensor(coo).to(device)
        v = torch.FloatTensor(np.ones(len(coo[0]))).to(device)

        X_batch = torch.sparse.FloatTensor(i, v, torch.Size([batch_size, nb_steps, nb_units])).to(device, dtype)
        y_batch = torch.tensor(labels_[batch_index], device=device)

        yield X_batch, y_batch

        counter += 1

#=======================================================================================

def plot_spk_rec(spk_rec, idx, plot_name="Spikes record"):

    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))+1
    gs = GridSpec(d,d)
    fig= plt.figure(figsize=(30,20),dpi=150)
    for i in range(nb_plt):
        plt.subplot(gs[i])
        plt.imshow(spk_rec[idx[i]].T,cmap=plt.cm.gray_r, origin="lower", aspect='auto')
        if i==0:
            plt.title(plot_name)
            plt.xlabel("Time")
            plt.ylabel("Units")    


def plot_mem_rec(mem, idx, plot_name="Membrane potential record"):
    
    nb_plt = len(idx)
    d = int(np.sqrt(nb_plt))+1
    dim = (d, d)
    
    gs=GridSpec(*dim)
    plt.figure(figsize=(30,20))
    dat = mem[idx]

    for i in range(nb_plt):
        if i==0:
            a0=ax=plt.subplot(gs[i])
            plt.title(plot_name)
        else: ax=plt.subplot(gs[i],sharey=a0)
        ax.plot(dat[i])


def plot_model_output(model, User_params, dataloader, device, dtype):
    if User_params['Project'] == "Image":
        X_batch, _ = next(iter(dataloader))
        X_batch = X_batch.to(device, dtype)
        X_batch = torch.reshape(X_batch, (User_params['batch_size'], User_params['Input_channels'], User_params['image_H'], User_params['image_W']))
    elif User_params['Project'] == "VPN":
        X_batch = next(iter(dataloader))[0].to(device, dtype)
    model(X_batch)

    nb_plt = 9
    if (User_params['batch_size'] >= 64):
        batch_idx = np.random.choice(User_params['batch_size'], nb_plt, replace=False)
    else:
        batch_idx = np.random.choice(User_params['batch_size'], nb_plt, replace=True)

    # Plotting spike trains or membrane potential
    for i, l in enumerate(model.layers):

        if isinstance(l, models.SpikingDenseLayer):
            print("Layer {}: average number of spikes={:.4f}".format(i, User_params['nb_steps'] * l.spk_rec_hist.mean()))
            spk_rec = l.spk_rec_hist
            if (User_params['Plot_graphs']):
                plot_spk_rec(spk_rec, idx=batch_idx, plot_name="Spikes record of SpikingDenseLayer")
        elif isinstance(l, models.SpikingConv2DLayer):
            print("Layer {}: average number of spikes={:.4f}".format(i, User_params['nb_steps'] * l.spk_rec_hist.mean()))
            spk_rec = l.spk_rec_hist
            if (User_params['Plot_graphs']):
                plot_spk_rec(spk_rec.sum(1), idx=batch_idx, plot_name="Spikes record of SpikingConv2DLayer")
        elif isinstance(l, models.SpikingConv2DLayer_custom):
            print("Layer {}: average number of spikes={:.4f}".format(i, User_params['nb_steps'] * l.spk_rec_hist.mean()))
            spk_rec = l.spk_rec_hist
            spk_rec = spk_rec.sum(1)
            spk_rec = np.reshape(spk_rec, (
            User_params['batch_size'], User_params['nb_steps'], l.output_shape[0] * l.output_shape[1]))
            if (User_params['Plot_graphs']):
                plot_spk_rec(spk_rec, idx=batch_idx, plot_name="Spikes record of SpikingConv2DLayer_custom")
        elif isinstance(l, models.SpikingConv3DLayer):
            print("Layer {}: average number of spikes={:.4f}".format(i, User_params['nb_steps'] * l.spk_rec_hist.mean()))
            spk_rec = l.spk_rec_hist
            spk_rec = spk_rec.sum(1)
            spk_rec = np.reshape(spk_rec, (
            User_params['batch_size'], User_params['nb_steps'], l.output_shape[0] * l.output_shape[1]))
            if (User_params['Plot_graphs']):
                plot_spk_rec(spk_rec, idx=batch_idx, plot_name="Spikes record of SpikingConv3DLayer")
        elif isinstance(l, models.SpikingConv3DLayer_separable):
            print("Layer {}: average number of spikes={:.4f}".format(i, User_params['nb_steps'] * l.spk_rec_hist.mean()))
            spk_rec = l.spk_rec_hist
            spk_rec = spk_rec.sum(1)
            spk_rec = np.reshape(spk_rec, (
            User_params['batch_size'], User_params['nb_steps'], l.output_shape[0] * l.output_shape[1]))
            if (User_params['Plot_graphs']):
                plot_spk_rec(spk_rec, idx=batch_idx, plot_name="Spikes record of SpikingConv3DLayer_separable")
        elif isinstance(l, models.Spiking3DPoolingLayer):
            print("Layer {}: average number of spikes={:.4f}".format(i, User_params['nb_steps'] * l.spk_rec_hist.mean()))
            spk_rec = l.spk_rec_hist
            spk_rec = spk_rec.sum(1)
            spk_rec = np.reshape(spk_rec, (
            User_params['batch_size'], User_params['nb_steps'], l.output_shape[0] * l.output_shape[1]))
            if (User_params['Plot_graphs']):
                plot_spk_rec(spk_rec, idx=batch_idx, plot_name="Spikes record of Spiking3DPoolingLayer")
        elif models.isReadoutlayer(l):
            mem_rec = l.mem_rec_hist
            mthr = l.mthr_hist
            if (User_params['Plot_graphs']):
                start_step = User_params['Train_params']['Readout_steps_start']
                end_step = User_params['Train_params']['Readout_steps_end']
                plot_mem_rec(mem_rec[:,start_step:end_step,:], batch_idx, plot_name="Membrane potential record of ReadoutLayer")
                plot_mem_rec(mthr[:,start_step:end_step,:], batch_idx, plot_name="mthr potential record of ReadoutLayer")


    # for i, layer in enumerate(model.layers) :
        # if isinstance(layer, models.SpikingConv2DLayer_custom):
            # l = layer
    # mem_rec = l.mem_rec_hist
    # spk_rec = l.spk_rec_hist

    # batch_idx = list(range(l.out_channels))
    # mem = np.reshape(mem_rec[0, :, :, int(l.output_shape[0] / 2), int(l.output_shape[1] / 2)],
                     # (l.out_channels, User_params['nb_steps'], 1))
    # plot_mem_rec(mem, batch_idx,
                 # "membrane record of all the channels of the central position (x=14, y=14) for one image")

    # batch_idx = list(range((int(l.output_shape[1] / 2) + 1)))
    # mem = np.reshape(
        # mem_rec[0, int(l.out_channels / 2), :, int(l.output_shape[0] / 2), :(int(l.output_shape[1] / 2) + 1)],
        # (User_params['nb_steps'], (int(l.output_shape[1] / 2) + 1)))
    # mem = (mem.transpose()).reshape((int(l.output_shape[1] / 2) + 1), User_params['nb_steps'], 1)
    # plot_mem_rec(mem, batch_idx,
                 # "membrane record of one channel, a column of position (x=14, y=0, … , 14 ) for one image")

    # batch_idx = list(range(User_params['batch_size']))
    # mem = np.reshape(mem_rec[:, int(l.out_channels / 2), :, int(l.output_shape[0] / 2), int(l.output_shape[1] / 2)],
                     # (User_params['batch_size'], User_params['nb_steps'], 1))
    # plot_mem_rec(mem, batch_idx, "membrane record of one channel of the central positon for all the images in a batch")

    # spk_rec_sum = spk_rec[:, :, 3:, :, :]
    # spk_rec_sum = spk_rec.sum(2)
    # print("Maximum number of spikes in SpikingConv2DLayer_custom : ", np.max(spk_rec.sum(2)))
    # mem_rec_plot = []
    # for i in range(User_params['batch_size']):
        # for c in range(l.out_channels):
            # for x in range(l.output_shape[0]):
                # for y in range(l.output_shape[1]):
                    # if (spk_rec_sum[i, c, x, y] > 0):
                        # mem_rec_plot.append(np.reshape(mem_rec[i, c, :, x, y], (User_params['nb_steps'], 1)))
    # mem_rec_plot = np.array(mem_rec_plot)
    # batch_idx = np.random.choice(mem_rec_plot.shape[0], 16, replace=False)
    # plot_mem_rec(mem_rec_plot, batch_idx, "membrane record of neurons with spike in output")


def plot_loss_acc_epoch_charts(User_params, checkpoint) :
    try :
        Epoch_loss_list = checkpoint['Epoch_loss_list']
        Epoch_valid_accuracy_list = checkpoint['Epoch_valid_accuracy_list']
        Train_loss_list = checkpoint['Train_loss_list']
        
        Epoch_loss_list.insert(0, Train_loss_list[0])
        Epoch_valid_accuracy_list.insert(0, 0.1)
    except :
        Epoch_loss_list = []
        Epoch_valid_accuracy_list = []
        Train_loss_list = []

    if User_params['Project'] == "VPN" :
        try :
            Epoch_average_precision_list = checkpoint['Epoch_average_precision_list']
            Epoch_average_precision_list.insert(0, 0.1)
        except :
            Epoch_average_precision_list = []

    loss_epochs = np.arange(len(Epoch_loss_list))
    acc_epochs = np.arange(len(Epoch_valid_accuracy_list))
    average_precision_epochs = np.arange(len(Epoch_average_precision_list))
    loss_batches = np.arange(len(Train_loss_list))

    plt.figure()
    # plt.subplot(411)
    plt.title("Loss vs Epochs chart")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.plot(loss_epochs, Epoch_loss_list)
    print("Epoch_loss_list: ", np.round(Epoch_loss_list,3))

    plt.figure()
    # plt.subplot(412)
    plt.title("Validation accuracy vs Epochs chart")
    plt.xlabel("Epochs")
    plt.ylabel("Validation Accuracy")
    plt.plot(acc_epochs, Epoch_valid_accuracy_list)
    print("Epoch_valid_accuracy_list: ", np.round(Epoch_valid_accuracy_list,3))

    plt.figure()
    # plt.subplot(413)
    plt.title("Loss vs Batches chart")
    plt.xlabel("Batches")
    plt.ylabel("Loss")
    plt.plot(loss_batches, Train_loss_list)
    
    if User_params['Project'] == "VPN" :
        plt.figure()
        # plt.subplot(414)
        plt.title("average_precision vs Epochs chart")
        plt.xlabel("Epochs")
        plt.ylabel("average_precision")
        plt.plot(average_precision_epochs, Epoch_average_precision_list)
    
    
def print_model_output(model, User_params, data_loader, device, dtype) :
    if User_params['Project'] == "Image":
        X_batch, _ = next(iter(data_loader))
        X_batch = X_batch.to(device, dtype)
        X_batch = torch.reshape(X_batch, (User_params['batch_size'], User_params['Input_channels'], User_params['image_H'], User_params['image_W']))
    elif User_params['Project'] == "VPN":
        X_batch = next(iter(data_loader))[0].to(device, dtype)
    model(X_batch)

    for i, l in enumerate(model.layers):
        if not models.isReadoutlayer(l):
            print(
                "Layer {}: average number of spikes={:.4f}".format(i, User_params['nb_steps'] * l.spk_rec_hist.mean()))

#=======================================================================================

def save_results(model, checkpoint, User_params, Results_path):
    try :
        with open(Results_path, 'r', encoding='utf-8') as json_file:
            results = json.load(json_file)
    except :
        results = {}
        # results['next_index'] = 1
    # next_index = results['next_index']
    next_index = str(checkpoint['model_id'])
    results[next_index] = {}

    results[next_index]['model_id'] = str(checkpoint['model_id'])
    results[next_index]['date_time'] = str(datetime.datetime.now())
    results[next_index]['time_zone'] = str(datetime.datetime.now(datetime.timezone.utc).astimezone().tzname())
    results[next_index]['model_layers'] = str(model.layers)
    results[next_index]['model_layers_params'] = {}

    for i, l in enumerate(model.layers):
        results[next_index]['model_layers_params'][i] = {}
        for name, value in vars(l).items() :
            if name[0] != '_' and name != 'spk_rec_hist' and name != 'mem_rec_hist' and name != 'mthr_hist' and name!='mask' and name != 'reset_parameters' and name != 'clamp' and name!='spike_fn' and name!='threshold':
                results[next_index]['model_layers_params'][i][str(name)] = (value)
        # for attr in dir(l) :
        #     if not callable(getattr(l, attr)) and not attr.startswith("_") :

    results[next_index]['epoch'] = checkpoint['epoch']
    results[next_index]['Train_Batch_Num'] = checkpoint['Batch_Num']
    results[next_index]['Epoch_loss_list'] = checkpoint['Epoch_loss_list']
    results[next_index]['Epoch_valid_accuracy_list'] = checkpoint['Epoch_valid_accuracy_list']
    results[next_index]['epoch_train_time'] = checkpoint['epoch_train_time']
    results[next_index]['validation_time'] = checkpoint['validation_time']
    try :
        results[next_index]['test_accuracy'] = checkpoint['test_accuracy']
    except :
        results[next_index]['test_accuracy'] = "Not tested!"
    try :
        results[next_index]['train_accuracy'] = checkpoint['train_accuracy']
    except :
        results[next_index]['train_accuracy'] = "Not tested!"

    results[next_index]['User_params'] = User_params
    # results['next_index'] = results['next_index'] + 1

    with open(Results_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=4)


def json_exist(model_id, Results_path):
    try :
        with open(Results_path, 'r', encoding='utf-8') as json_file:
            results = json.load(json_file)
    except :
        return False
    try :
        results[str(model_id)]
        return True
    except :
        return False


def bubblesort(list):
# Swap the elements to arrange in order
    for iter_num in range(len(list)-1,0,-1):
        for idx in range(iter_num):
            if list[idx]>list[idx+1]:
                temp = list[idx]
                list[idx] = list[idx+1]
                list[idx+1] = temp

def find_model(User_params, Find_path) :
    try :
        with open(Find_path, 'r', encoding='utf-8-sig') as json_file:
            results = json.load(json_file)
    except :
        results = {}
        print("No results exist!!!")

    if User_params['find_params']['Find_mode'] == "similar_model":
        main_mismatch = {}
        non_main_mismatch = {}
        main_no_exist = {}
        non_main_no_exist = {}
        results_penalty = {}
        penalty_list=[]
        results_id_list=[]

        main_params = User_params['find_params']['main_params']
        dont_care_params = User_params['find_params']['dont_care_params']
        main_mismatch_penalty = User_params['find_params']['main_mismatch_penalty']
        non_main_mismatch_penalty = User_params['find_params']['non_main_mismatch_penalty']
        main_no_exist_penalty = User_params['find_params']['main_no_exist_penalty']
        non_main_no_exist_penalty = User_params['find_params']['non_main_no_exist_penalty']

        for result in results :
            main_mismatch[result] = []
            non_main_mismatch[result] = []
            main_no_exist[result] = []
            non_main_no_exist[result] = []

            for param in User_params :
                param_exists = True
                if not param in dont_care_params :
                    if 'User_params' in results[result] :
                        if param in results[result]['User_params']:
                            model_param = results[result]['User_params'][param]
                        else :
                            param_exists = False
                    else :
                        if param in results[result]:
                            model_param = results[result][param]
                        else :
                            param_exists = False

                    if param_exists :
                        if not model_param == User_params[param]:
                            if param in main_params:
                                main_mismatch[result].append(param)
                            else:
                                non_main_mismatch[result].append(param)
                    else :
                        if param in main_params :
                            main_no_exist[result].append(param)
                        else :
                            non_main_no_exist[result].append(param)
                else :
                    continue

        for result in results:
            results_penalty[result] = len(main_mismatch[result])*main_mismatch_penalty + len(non_main_mismatch[result])*non_main_mismatch_penalty \
                                    + len(main_no_exist[result])*main_no_exist_penalty + len(non_main_no_exist[result])*non_main_no_exist_penalty
            penalty_list.append(results_penalty[result])
            results_id_list.append(result)

        # Bubble sort algorithm
        for iter_num in range(len(penalty_list)-1,0,-1):
            for idx in range(iter_num):
                if penalty_list[idx]>penalty_list[idx+1]:
                    temp = penalty_list[idx]
                    penalty_list[idx] = penalty_list[idx+1]
                    penalty_list[idx+1] = temp

                    temp = results_id_list[idx]
                    results_id_list[idx] = results_id_list[idx + 1]
                    results_id_list[idx + 1] = temp

        print("result ID\t\t\t\t\t\t\t\t\tPenalty\tAccuracy")
        index = 0
        for result in results_id_list :
            print(str(index+1) + '-' + result + ' : ' + '\t' + str(penalty_list[index]) + '\t\t' + str(results[result]["Epoch_valid_accuracy_list"][-1]))

            if User_params['find_params']['Find_print_details']:
                print("main_mismatch for #%i:", index + 1)
                for param in main_mismatch[result] :
                    print(param)
                print("non_main_mismatch for #%i:", index + 1)
                for param in non_main_mismatch[result]:
                    print(param)
                print("main_no_exist for #%i:", index + 1)
                for param in main_no_exist[result]:
                    print(param)
                print("non_main_no_exist for #%i:", index + 1)
                for param in non_main_no_exist[result]:
                    print(param)
            index+=1

    elif User_params['find_params']['Find_mode'] == "max_accuracy":
        results_id_list = []
        accuracy_list = []
        for result in results:
            results_id_list.append(result)
            accuracy_list.append(results[result]["Epoch_valid_accuracy_list"][-1])

        for iter_num in range(len(accuracy_list)-1,0,-1):
            for idx in range(iter_num):
                if accuracy_list[idx]>accuracy_list[idx+1]:
                    temp = accuracy_list[idx]
                    accuracy_list[idx] = accuracy_list[idx+1]
                    accuracy_list[idx+1] = temp

                    temp = results_id_list[idx]
                    results_id_list[idx] = results_id_list[idx + 1]
                    results_id_list[idx + 1] = temp


        print("result ID\t\t\t\t\t\t\t\t\tAccuracy")
        results_len = len(results_id_list)
        for index in range(len(accuracy_list)) :
            print(str(index+1) + '-' + results_id_list[results_len-index-1] + ' : ' + '\t' + str(accuracy_list[results_len-index-1]))