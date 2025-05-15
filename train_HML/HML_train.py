import os
import sys
import numpy as np
import time
import yaml
import matplotlib.pyplot as plt
import importlib
import torch
from torch.optim.lr_scheduler import ExponentialLR
from torch.utils.data import TensorDataset, DataLoader, random_split
from parflow.tools.io import write_pfb, read_pfb
from helper_modules import early_stopper, weight_init
import subsettools as st



#########################################################################################################
####################################Generate Dataset#####################################################
#########################################################################################################
def init_data():
    ticc = time.time()
    print("start loading data")

    slope_x = read_pfb('../data/slope_x.pfb')
    slope_y = read_pfb('../data/slope_y.pfb')
    k = read_pfb('../data/k.pfb')
    pme = read_pfb('../data/pme.pfb')
    WTD = np.load('../data/wtdepth3.npy')[2:-1, 1:-1]
    mask = read_pfb('../data/mask.pfb').squeeze()

    toc = time.time()
    print("all data is loadad in time =", np.round(toc-ticc,3))

    pme=pme[-1]*mask
    WTD = WTD*mask
    slope_x = slope_x.squeeze()*mask
    slope_y = slope_y.squeeze()*mask

    #make k 2-dimensional: calculate geometric means of K
    thickness_list = [200.0,100.0,50.0,25.0,10.0,5.0,1.0,0.6,0.3,0.1] #Thickness of each layer
    multi = 1
    for i in range(10):
        multi += (thickness_list[i]/k[i,:,:])
    k_2d = 392.0 / multi 
    k_2d[k_2d==0] = np.nan
    k_2d=k_2d*mask


    # normalize
    pme_min = np.nanmin(pme)
    pme_max = np.nanmax(pme)
    if input_shape_prints_bool: print("all_PmE.shape = ",pme.shape)
    pme_norm = (pme - pme_min) / (pme_max - pme_min)

    ksat_min = np.nanmin(k_2d)
    ksat_max = np.nanmax(k_2d)
    if input_shape_prints_bool: print("all_k.shape = ",k.shape)
    k_norm = (k_2d - ksat_min) / (ksat_max - ksat_min)

    slope_x_min = np.nanmin(slope_x)
    slope_x_max = np.nanmax(slope_x)
    if input_shape_prints_bool: print("all_slope_x.shape = ",slope_x.shape)
    slopes_x_norm = (slope_x - slope_x_min) / (slope_x_max - slope_x_min)

    slope_y_min = np.nanmin(slope_y)
    slope_y_max = np.nanmax(slope_y)
    if input_shape_prints_bool: print("all_slope_y.shape = ",slope_y.shape)
    slopes_y_norm = (slope_y - slope_y_min) / (slope_y_max - slope_y_min)

    WTD_min = np.nanmin(WTD)
    WTD_max = np.nanmax(WTD)

    all_minmaxs = np.array([[WTD_min, WTD_max], [pme_min, pme_max], [ksat_min, ksat_max], [slope_x_min, slope_x_max], [slope_y_min, slope_y_max]])
    if input_shape_prints_bool: print("all_minmaxs = ",all_minmaxs)
    torch.save(torch.tensor(all_minmaxs), 'test_data/' + saving_nr_string + '_minmaxs.csv')

    # generate 150x150 samples for training
    step_x = 50
    step_y = 50

    curr_x = 0
    curr_y = 0

    WTD_y_shape = WTD.shape[0]
    WTD_x_shape = WTD.shape[1]

    all_WTD = []
    all_PmE_norm = []
    all_k_norm = []
    all_slopes_x_norm = []
    all_slopes_y_norm = []

    #demo basins
    testing_basins = ['11020002','14070003', '11090101', '14070001', '07040003']

    exclude_bounds = []
    grid = "conus2"
    for curr_huc in testing_basins:
        curr_huc_list = [curr_huc]
        ij_bounds, _ = st.define_huc_domain(hucs=curr_huc_list, grid=grid)
        exclude_bounds.append(ij_bounds)

    all_used_ones = []
    all_excluded_ones = []
    how_often = 0

    while curr_y + 150 < WTD_y_shape:
        how_often += 1
        if curr_x + 150 < WTD_x_shape:
            exclude_this = False
            for curr_exclude in exclude_bounds:
                if curr_x <= curr_exclude[0] and curr_x +150 >= curr_exclude[0]:
                    if curr_y <= curr_exclude[1] and curr_y+150 >= curr_exclude[1]:
                        exclude_this = True
                    elif curr_y >= curr_exclude[1] and curr_y <= curr_exclude[3]:
                        exclude_this = True
                elif curr_x >= curr_exclude[0] and curr_x <= curr_exclude[2]:
                    if curr_y <= curr_exclude[1] and curr_y+150 >= curr_exclude[1]:
                        exclude_this = True
                    elif curr_y >= curr_exclude[1] and curr_y <= curr_exclude[3]:
                        exclude_this = True

            if exclude_this:
                all_excluded_ones.append([curr_x, curr_y])
                curr_x += step_x
            else:
                selected_region = mask[curr_y:curr_y+150, curr_x:curr_x+150]
                num_nonzeros = np.count_nonzero(selected_region)
                if num_nonzeros <= (selected_region.size // 2):
                    curr_x += step_x
                    all_excluded_ones.append([curr_x, curr_y])            
                else:
                    all_WTD.append(WTD[curr_y:curr_y+150, curr_x:curr_x+150])
                    all_PmE_norm.append(pme_norm[curr_y:curr_y+150, curr_x:curr_x+150])
                    all_k_norm.append(k_norm[curr_y:curr_y+150, curr_x:curr_x+150])
                    all_slopes_x_norm.append(slopes_x_norm[curr_y:curr_y+150, curr_x:curr_x+150])
                    all_slopes_y_norm.append(slopes_y_norm[curr_y:curr_y+150, curr_x:curr_x+150])
                    all_used_ones.append([curr_x, curr_y])
                    curr_x += step_x
        else:
            curr_x = 0
            curr_y += step_y
    if input_shape_prints_bool: print('I loop through the data', how_often, 'times')
    if input_shape_prints_bool: print('I have', len(all_WTD), 'samples')

    all_WTD = np.array(all_WTD)
    all_PmE_norm = np.array(all_PmE_norm)
    all_k_norm = np.array(all_k_norm)
    all_slopes_x_norm = np.array(all_slopes_x_norm)
    all_slopes_y_norm = np.array(all_slopes_y_norm)

    #change all nans to zeros
    if input_shape_prints_bool: print("all_WTD.shape before no_nan = ",all_WTD.shape)
    all_WTD_no_nan = np.nan_to_num(all_WTD, nan=0.0)
    if input_shape_prints_bool: print("all_WTD_no_nan.shape = ",all_WTD_no_nan.shape)
    all_PmE_norm_no_nan = np.nan_to_num(all_PmE_norm, nan=0.0)
    all_k_norm_no_nan = np.nan_to_num(all_k_norm, nan=0.0)
    all_slopes_x_norm_no_nan = np.nan_to_num(all_slopes_x_norm, nan=0.0)
    all_slopes_y_norm_no_nan = np.nan_to_num(all_slopes_y_norm, nan=0.0)

    # Compute and save the mean of all WTDs
    all_WTD_norm = (all_WTD_no_nan - WTD_min) / (WTD_max - WTD_min)
    all_WTD_mean_norm = np.mean(all_WTD_norm, axis=0)
    WTD_mean_saving_name = 'test_data/' + saving_nr_string + '_all_WTD_mean.csv'
    torch.save(all_WTD_mean_norm, WTD_mean_saving_name)


    #define the seed 
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

    #########################################################################################################
    ####################################Load Model###########################################################
    #########################################################################################################
    mldef = importlib.import_module(ml_model)
    if input_shape_prints_bool: print("Using model:")
    if input_shape_prints_bool: print(mldef)
    model = mldef.my_model(verbose = verbose_bool, use_dropout = use_dropout_bool, grid_size=[n_dims1, n_dims2], seed = seed)
    model.to(cuda)

    return  model, all_WTD_no_nan, all_PmE_norm_no_nan, all_k_norm_no_nan, all_slopes_x_norm_no_nan, all_slopes_y_norm_no_nan, all_used_ones

#train the model
def train_model(model, all_WTD_no_nan, all_PmE_norm_no_nan, all_k_norm_no_nan, all_slopes_x_norm_no_nan, all_slopes_y_norm_no_nan, all_used_ones):
    weight_initializer = weight_init(seed = seed)
    model.apply(weight_initializer.init_weights)

    #define the optimizer and the early stopping
    optim = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=optimizer_momentum, nesterov = nesterov_bool)
    optim.zero_grad()

    if early_stopping_bool:
        my_early_stopper = early_stopper(wait=early_stopper_wait, acceptable_diff=early_stopper_acc_diff)

    if lr_scheduler_bool:
        scheduler = ExponentialLR(optim, gamma=lr_gamma)

    torch.save(n_epochs, 'test_data/' + saving_nr_string + '_nr_epochs.csv')

    #########################################################################################################
    ####################################Data Management######################################################
    #########################################################################################################

    # exclude the eight most atypical cases
    WTD_mean = np.mean(all_WTD_no_nan, axis=0)
    WTD_deviations_1 = all_WTD_no_nan - WTD_mean
    WTD_deviation_norm = np.linalg.norm(WTD_deviations_1, axis=(1, 2))
    top_indices = np.argsort(WTD_deviation_norm)[-8:]
    if input_shape_prints_bool: print('Most atypical WTD deviations = ',WTD_deviation_norm[top_indices])
    if input_shape_prints_bool: print("(y,x) of the 8 WTDs that deviate the most:", [all_used_ones[i] for i in top_indices])

    extreme_WTD = all_WTD_no_nan[top_indices]
    WTD_filtered = np.delete(all_WTD_no_nan, top_indices, axis=0)
    

    extreme_PmE_norm = all_PmE_norm_no_nan[top_indices]
    PmE_norm_filtered = np.delete(all_PmE_norm_no_nan, top_indices, axis=0)

    extreme_Ksat_norm = all_k_norm_no_nan[top_indices]
    Ksat_norm_filtered = np.delete(all_k_norm_no_nan, top_indices, axis=0)

    extreme_slope_x_norm = all_slopes_x_norm_no_nan[top_indices]
    slope_x_norm_filtered = np.delete(all_slopes_x_norm_no_nan, top_indices, axis=0)

    extreme_slope_y_norm = all_slopes_y_norm_no_nan[top_indices]
    slope_y_norm_filtered = np.delete(all_slopes_y_norm_no_nan, top_indices, axis=0)

    #create the atypical dataset
    np_extreme_data  = np.vstack(([extreme_Ksat_norm], [extreme_PmE_norm], [extreme_slope_x_norm], [extreme_slope_y_norm]))
    #permutation is necessary such that the first datasample consists of one of each features
    extreme_data_orig = torch.from_numpy(np_extreme_data).clone().detach().to(torch.float32).permute(1,0,2,3)
    extreme_label_orig = extreme_WTD

    #save extreme test features
    testindicessavingName_extreme = 'test_data/' + saving_nr_string + '_test_indices_extreme.csv'
    torch.save(top_indices, testindicessavingName_extreme)
    testfeaturesavingName_extreme = 'test_data/' + saving_nr_string + '_test_features_extreme.csv'
    torch.save(extreme_data_orig, testfeaturesavingName_extreme)
    testlabelsavingName_extreme = 'test_data/' + saving_nr_string + '_test_labels_extreme.csv'
    torch.save(extreme_label_orig, testlabelsavingName_extreme)

    #create the training, validation, testing dataset
    np_data = np.vstack(([Ksat_norm_filtered], [PmE_norm_filtered], [slope_x_norm_filtered], [slope_y_norm_filtered]))
    #permutation is necessary such that the first datasample consists of one of each features
    data_orig = torch.from_numpy(np_data).clone().detach().to(torch.float32).permute(1,0,2,3)
    if input_shape_prints_bool: print('data_orig.shape = ',data_orig.shape)
    label_orig = torch.tensor(WTD_filtered, dtype = torch.float32).unsqueeze(1)
    if input_shape_prints_bool: print('label_orig.shape = ',label_orig.shape)

    data_orig_dataset = TensorDataset(data_orig, label_orig)

    # Define the sizes of train, test, and validation sets
    train_size = int(train_data_ratio * len(data_orig_dataset))
    test_size = int(test_data_ratio * len(data_orig_dataset))
    vali_size = len(data_orig_dataset) - train_size - test_size
    if input_shape_prints_bool: print('train_size = ',train_size)
    if input_shape_prints_bool: print('test_size = ',test_size)
    if input_shape_prints_bool: print('vali_size = ',vali_size)

    # Split the dataset into train, test, and validation sets
    train_data_all_ss, test_data_ss, vali_data_ss = random_split(data_orig_dataset, [train_size, test_size, vali_size])

    train_loader_all = DataLoader(train_data_all_ss, batch_size=len(train_data_all_ss))
    test_loader = DataLoader(test_data_ss, batch_size=len(test_data_ss))
    vali_loader = DataLoader(vali_data_ss, batch_size=mini_batch_size_vali)
    
    train_data_all, train_label_all = next(iter(train_loader_all))
    test_data, test_label = next(iter(test_loader))

    train_dataset = TensorDataset(train_data_all, train_label_all)

    if input_shape_prints_bool: print("train label.shape = ",train_label_all.shape)
    if input_shape_prints_bool: print("train data.shape = ",train_data_all.shape)
    if input_shape_prints_bool: print("test label.shape = ",test_label.shape)
    if input_shape_prints_bool: print("test_data.shape = ",test_data.shape)

    #save test features
    testindicessavingName = 'test_data/' + saving_nr_string + '_test_indices.csv'
    torch.save(test_data_ss.indices, testindicessavingName)
    testfeaturesavingName = 'test_data/' + saving_nr_string + '_test_features.csv'
    torch.save(test_data, testfeaturesavingName)
    testlabelsavingName = 'test_data/' + saving_nr_string + '_test_labels.csv'
    torch.save(test_label, testlabelsavingName)

    all_train_losses = []
    all_vali_losses = []

    train_loader = DataLoader(train_dataset, batch_size=mini_batch_size_train, shuffle = True)

    #########################################################################################################
    ####################################Model Training######################################################
    #########################################################################################################

    #train the model
    for epoch in range(n_epochs):
        if input_shape_prints_bool: print('\n here in epoch,', epoch)
        model.train()
        counter = 0
        sum_train_loss = 0
        for i, (train_features, train_labels) in enumerate(train_loader):
            j = 0

            for k, data in enumerate(train_features, 0):
                output = train_labels[k]
                label = output.to(cuda)
                input = data.to(cuda)

                if torch.isnan(input).any() or torch.isnan(label).any():
                    print('NaN detected in input or label')
                    continue

                optim.zero_grad()

                #compute backwards propagation
                pred = model(input) #forward pass
                loss = model.loss(pred, label)
                if input_shape_prints_bool: print('current loss :' + str(loss.item()))
                loss.backward()
                if input_shape_prints_bool: print('backward propagation completed after epoch', epoch, 'in i=', i)

                sum_train_loss += loss.data

                #do a gradient step every accumulation_steps
                if ((j+1) % accumulation_steps == 0):
                    optim.step()
                    optim.zero_grad()
                j += 1
                counter += 1

            #free up CUDA memory
            input = None
            label = None
            torch.cuda.empty_cache()
        
        #evaluate training loss
        train_loss = 1/(counter) * sum_train_loss
        all_train_losses.append(train_loss)
        print("final train_loss = ",train_loss)

        # evaluate validation loss
        with torch.no_grad():
            model.eval()
            sum_vali_loss = 0
            counter = 0
            for i, (features_vali, labels_vali) in enumerate(vali_loader):
            
                for k, curr_vali_data in enumerate(features_vali, 0):
                    vali_output = labels_vali[k]
                    vali_input_cuda = curr_vali_data.to(cuda)
                    vali_label_cuda = vali_output.to(cuda)

                    if torch.isnan(vali_input_cuda).any() or torch.isnan(vali_label_cuda).any():
                        print('NaN detected in input or label')
                        continue

                    vali_pred = model(vali_input_cuda)
                    loss = model.loss(vali_pred, vali_label_cuda)
                    sum_vali_loss += loss.data
                    counter += 1
                
                #free up CUDA memory
                vali_input_cuda = None
                vali_label_cuda = None
                torch.cuda.empty_cache()

            vali_loss = 1/(counter) * sum_vali_loss
        
        all_vali_losses.append(vali_loss)
        if input_shape_prints_bool: print("final vali_loss = ",vali_loss)	

        #activate early stopping
        if early_stopping_bool:
            if epoch >= 100 and my_early_stopper.early_stop(vali_loss):
                print("Early stopping activated")
                torch.save(epoch, 'test_data/' + saving_nr_string + '_nr_epochs.csv')
                break

        #step the learning rate scheduler
        if lr_scheduler_bool:
            scheduler.step()
        
    #convert losses to cpu
    cpu_train_losses = [loss.cpu().detach().numpy() for loss in all_train_losses]
    cpu_vali_losses = [loss.cpu().detach().numpy() for loss in all_vali_losses]
    all_train_losses = None
    all_vali_losses = None
    np.savetxt("RMSE_losses/" + saving_nr_string + "_train_losses", cpu_train_losses, delimiter = ",")
    np.savetxt("RMSE_losses/" + saving_nr_string + "_vali_losses", cpu_vali_losses, delimiter = ",")

    #save trained model
    mdlsavingName = 'models/' + model_saving_name + '_my_trained_model.pth'
    torch.save(model.state_dict(), mdlsavingName)

#########################################################################################################
####################################Plot RMSEs during training###########################################
#########################################################################################################
def plot_MSE_dev():
    from matplotlib.gridspec import GridSpec

    all_train_losses = np.loadtxt("RMSE_losses/" + saving_nr_string + "_train_losses", delimiter=",")
    all_vali_losses = np.loadtxt("RMSE_losses/" + saving_nr_string + "_vali_losses", delimiter=",")
    max_nr_epochs = torch.load('test_data/' + saving_nr_string + '_nr_epochs.csv')

    fig = plt.figure(figsize=(8, 6))
    gs = GridSpec(1, 2, width_ratios=[3, 1], wspace=0.1)

    ax1 = fig.add_subplot(gs[0])
    ax2 = fig.add_subplot(gs[1], sharey=ax1)

    epochs_array = np.arange(1, len(all_train_losses) + 1)

    # Plot on the first subplot
    ax1.plot(epochs_array, np.sqrt(all_train_losses), label="Training loss")
    ax1.plot(epochs_array, np.sqrt(all_vali_losses), label="Validation loss")
    ax1.set_xlim(0, 80)
    ax1.set_ylabel("RMSE", fontsize=14)
    ax1.set_yscale("log")
    ax1.set_title("Loss development", fontsize=14)

    # Plot on the second subplot
    ax2.plot(epochs_array, np.sqrt(all_train_losses), label="Training loss")
    ax2.plot(epochs_array, np.sqrt(all_vali_losses), label="Validation loss")
    ax2.set_xlim(270, max_nr_epochs)
    ax2.set_yscale("log")

    # Remove spines between ax1 and ax2
    ax2.spines['left'].set_visible(False)
    ax1.spines['right'].set_visible(False)
    ax2.yaxis.tick_right()
    ax2.tick_params(labelright=False)
    fig.subplots_adjust(wspace=0.05)

    # Add diagonal lines to indicate the break
    d = .015
    kwargs = dict(transform=ax1.transAxes, color='k', clip_on=False)
    ax1.plot((1-3*d, 1+3*d), (-3*d, +3*d), **kwargs)       # Top-left diagonal
    ax1.plot((1-d, 1+d), (1-d, 1+d), **kwargs)     # Bottom-left diagonal

    kwargs.update(transform=ax2.transAxes)
    ax2.plot((-d, +d), (1-d, 1+d), **kwargs)       # Top-right diagonal
    ax2.plot((-d, +d), (-d, +d), **kwargs)         # Bottom-right diagonal


    # Legends, ticks, etc.
    ax1.set_yticks([20, 100, 500])
    ax1.yaxis.set_tick_params(labelsize=14)
    ax1.set_xticks([0, 25, 50, 75])
    ax1.xaxis.set_tick_params(labelsize=14)
    ax2.set_xticks([275,300])
    ax2.xaxis.set_tick_params(labelsize=14)
    ax2.legend(loc='upper right', fontsize=14)

    # Add common x-label at the center
    fig.text(0.5, 0.04, 'Epoch', ha='center', fontsize=14)

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.savefig('plots/RMSE_losses/' + saving_nr_string + '_RMSE_development.png', dpi=400)
    plt.show()
    plt.close()

#########################################################################################################
####################################Compute final Test error#############################################
#########################################################################################################d
def compute_overall_test_loss(model):
    model.eval()
    mdlsavingName = 'models/' + model_saving_name + '_my_trained_model.pth'
    model.load_state_dict(torch.load(mdlsavingName))

    all_minmaxs = torch.load('test_data/' + saving_nr_string + '_minmaxs.csv')
    WTD_min = all_minmaxs[0,0]
    WTD_max = all_minmaxs[0,1]

    #load test data
    test_features = torch.load('test_data/' + saving_nr_string + '_test_features.csv')
    test_labels = torch.load('test_data/' + saving_nr_string + '_test_labels.csv')

    #load extreme test data
    test_features_extreme = torch.load('test_data/' + saving_nr_string + '_test_features_extreme.csv')
    test_labels_extreme = torch.load('test_data/' + saving_nr_string + '_test_labels_extreme.csv')

    #iterate over the test data
    torch.cuda.empty_cache()
    tic = time.time()
    with torch.no_grad():
        model.eval()
        sum_test_loss = 0
        for m, test_data in enumerate(test_features, 0):
            test_output = test_labels[m]
            test_input = test_data.to(cuda)
            test_label = test_output.to(cuda)
            test_pred = model(test_input)
            loss = model.loss(test_pred, test_label)
            sum_test_loss += loss.data
            test_input = None
            test_label = None
            
        test_loss = 1/(m+1) * sum_test_loss

        for n, test_data_extreme in enumerate(test_features_extreme, 0):
            test_output_extreme = torch.tensor(test_labels_extreme[n]).unsqueeze(0)
            test_input_extreme = test_data_extreme.to(cuda)
            test_label_extreme = test_output_extreme.to(cuda)
            test_pred_extreme = model(test_input_extreme)
            loss_extreme = model.loss(test_pred_extreme, test_label_extreme)
            sum_test_loss += loss_extreme.data
            test_input_extreme = None
            test_label_extreme = None

        toc = time.time()
        print("computation of test overall loss complete! Time = ", np.round(toc-tic,3))
        test_loss_w_extreme = 1/(m+n+1) * sum_test_loss

    overall_test_loss = test_loss.cpu().detach().numpy()
    overall_test_loss_w_extreme = test_loss_w_extreme.cpu().detach().numpy()

    print("overall_test_loss = ",overall_test_loss)
    print("overall_test_loss_w_extreme = ",overall_test_loss_w_extreme)

    return overall_test_loss, overall_test_loss_w_extreme

#########################################################################################################
####################################Evalaute Test Samples################################################
#########################################################################################################
def test_model(model, overall_test_loss, overall_test_loss_w_extreme, save_estims_bool, which_test_samples_factor):
    model.eval()
    mdlsavingName = 'models/' + model_saving_name + '_my_trained_model.pth'
    model.load_state_dict(torch.load(mdlsavingName))
    test_features = torch.load('test_data/' + saving_nr_string + '_test_features.csv')
    test_labels = torch.load('test_data/' + saving_nr_string + '_test_labels.csv')
    max_nr_epochs = torch.load('test_data/' + saving_nr_string + '_nr_epochs.csv')

    tic = time.time()
    for i, data in enumerate(test_features, 0):

        if i%which_test_samples_factor!=0:
            continue

        torch.cuda.empty_cache()
        label = test_labels[i]
        with torch.no_grad():
            input = data.to(cuda)
            res = model(input)
            curr_loss = model.loss(res, label.to(cuda)).data.cpu().detach().numpy()
            estim = res.float().cpu().detach().numpy()
            estim = estim.squeeze()

        label = label.squeeze()

        toc = time.time()
        print("computation and test evaluation complete! Overall time = ",np.round(toc-tic,3))

        if save_estims_bool:
            estimsavingName = 'DTWT_estims/' + saving_nr_string + '_' + str(i) +'_estim.csv'
            torch.save(estim, estimsavingName)

        fig, axes = plt.subplots(1,2, figsize=(20,10))
        axes[0].set_axis_off()
        im1 = axes[0].imshow(estim, cmap="twilight", origin='lower')

        if compute_overall_test_loss:
            axes[0].set_title("Estimation with RMSE = " + str(np.round(np.sqrt(curr_loss),5)) + "\
                              \n Average RMSE on all test samples (without/with extremes) = " + str(np.round(np.sqrt(overall_test_loss),5))+ "/"+str(np.round(np.sqrt(overall_test_loss_w_extreme),5))+" within " + str(max_nr_epochs) + " epochs.")
        else:
            axes[0].set_title("Estimation within " + str(max_nr_epochs) + " epochs")

        axes[1].set_axis_off()
        im2 = axes[1].imshow(label, cmap="twilight", origin='lower')
        axes[1].set_title("Ground truth")

        #normalize the colorbar, scale to same values
        vmin = min(im1.get_array().min(), im2.get_array().min())
        vmax = max(im1.get_array().max(), im2.get_array().max())
        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)

        # Create a colorbar for both plots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im2, cax=cbar_ax, orientation='vertical').set_label('DTWT in [m]', fontsize=12)
        cbar_ax.tick_params(labelsize=13)

        imgsavingName = 'plots/' + saving_nr_string + '_' + str(i) + '_estimation.png'
        plt.savefig(imgsavingName, dpi=1200)
        fig.show()
        plt.close()

        #free up CUDA memory
        input = None
        label = None

#########################################################################################################
#############################Evalaute Extreme Test Samples###############################################
#########################################################################################################
def test_model_extreme(model, overall_test_loss, overall_test_loss_w_extreme, save_extreme_estims_bool):
    model.eval()
    mdlsavingName = 'models/' + model_saving_name + '_my_trained_model.pth'
    model.load_state_dict(torch.load(mdlsavingName))
    max_nr_epochs = torch.load('test_data/' + saving_nr_string + '_nr_epochs.csv')
    extreme_indices = torch.load('test_data/' + saving_nr_string + '_test_indices_extreme.csv')

    test_features_extreme = torch.load('test_data/' + saving_nr_string + '_test_features_extreme.csv')
    test_labels_extreme = torch.load('test_data/' + saving_nr_string + '_test_labels_extreme.csv')
    test_features_extreme.to(cuda)

    tic = time.time()
    for i, data in enumerate(test_features_extreme, 0):
        torch.cuda.empty_cache()
        label = torch.tensor(test_labels_extreme[i]).unsqueeze(0)
        with torch.no_grad():
            input = data.to(cuda)
            res = model(input)
            curr_loss = model.loss(res, label.to(cuda)).data.cpu().detach().numpy()
            estim = res.float().cpu().detach().numpy().squeeze()

        label = label.squeeze()

        toc = time.time()
        print("Computation and evaluation on ", i,"'th extreme case complete! Overall time = ",np.round(toc-tic,3))
        
        if save_extreme_estims_bool:
            estimsavingName = 'DTWT_estims/' + saving_nr_string + '_' + str(i) +'_extreme_estim.csv'
            torch.save(estim, estimsavingName)

        fig, axes = plt.subplots(1,2, figsize=(20,10))
        axes[0].set_axis_off()
        im1 = axes[0].imshow(estim, cmap="twilight", origin='lower')

        if compute_overall_test_loss:
            axes[0].set_title("Estimation with RMSE = " + str(np.round(np.sqrt(curr_loss),5)) + "\
                              \n Average RMSE on all test samples (without/with extremes) = " + str(np.round(np.sqrt(overall_test_loss),5))+ "/"+str(np.round(np.sqrt(overall_test_loss_w_extreme),5))+" within " + str(max_nr_epochs) + " epochs.")
        else:
            axes[0].set_title("Estimation within " + str(max_nr_epochs) + " epochs. \n Extreme test case " + str(extreme_indices[i]))

        axes[1].set_axis_off()
        im2 = axes[1].imshow(label, cmap="twilight", origin='lower')
        axes[1].set_title("Ground truth")

        #normalize the colorbar, scale to same values
        vmin = min(im1.get_array().min(), im2.get_array().min())
        vmax = max(im1.get_array().max(), im2.get_array().max())
        im1.set_clim(vmin, vmax)
        im2.set_clim(vmin, vmax)

        # Create a colorbar for both plots
        cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
        fig.colorbar(im2, cax=cbar_ax, orientation='vertical').set_label('DTWT in [m]', fontsize=12)
        cbar_ax.tick_params(labelsize=13)

        imgsavingName = 'plots/' + saving_nr_string + '_' + str(i) + '_extreme_estim.png'
        plt.savefig(imgsavingName, dpi=1200)
        fig.show()
        plt.close()

        #free up CUDA memory
        input = None
        label = None


if __name__ == "__main__":
    #########################################################################################################
    ####################################Import USER settings#################################################
    #########################################################################################################

    config_file = sys.argv[1]
    print("Using config file:", config_file)

    file = open(config_file)
    settings = yaml.load(file, Loader=yaml.FullLoader)

    #load all parameters from the settings file
    saving_nr = settings['saving_nr']
    model_nr = settings['model_nr']

    train_model_bool = settings['train_model_bool']
    compute_overall_test_loss_bool = settings['compute_overall_test_loss_bool']
    test_model_bool = settings['test_model_bool']
    which_test_samples_factor = settings['which_test_samples_factor']
    test_model_extreme_bool = settings['test_model_extreme_bool']
    plot_MSE_dev_bool = settings['plot_MSE_dev_bool']
    save_estims_bool = settings['save_estims_bool']
    save_extreme_estims_bool = settings['save_extreme_estims_bool']
    input_shape_prints_bool = settings['input_shape_prints_bool']
    verbose_bool = settings['verbose_bool']

    lr_scheduler_bool = settings['lr_scheduler_bool']
    early_stopping_bool = settings['early_stopping_bool']
    early_stopper_wait = settings['early_stopper_wait']
    early_stopper_acc_diff = settings['early_stopper_acc_diff']
    use_dropout_bool = settings['use_dropout_bool']
    nesterov_bool = settings['nesterov_bool']

    ml_model = settings['ml_model']

    train_data_ratio = settings['train_data_ratio']
    test_data_ratio = settings['test_data_ratio']

    n_dims1 = settings['n_dims1']
    n_dims2 = settings['n_dims2']

    n_epochs = settings['n_epochs']
    seed = settings['seed']
    learning_rate = settings['learning_rate']
    lr_gamma = settings['lr_gamma']
    optimizer_momentum = settings['optimizer_momentum']

    amt_steps_per_sample = settings['amt_steps_per_sample']
    mini_batch_size_train = settings['mini_batch_size_train']
    mini_batch_size_vali = settings['mini_batch_size_vali']
    accumulation_steps = settings['accumulation_steps']

    # Create the run name
    saving_nr_string = saving_nr + "_sd" + str(seed) + "_" + str(n_epochs) + "_epochs"
    model_saving_name = model_nr + "_sd" + str(seed) + "_" + str(n_epochs) + "_epochs"

    #GPU
    print("torch.cuda.is_available() = ",torch.cuda.is_available())
    cuda = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cpu = torch.device('cpu')
    torch.cuda.empty_cache()


    model, all_WTD_no_nan, all_PmE_norm_no_nan, all_k_norm_no_nan, all_slopes_x_norm_no_nan, all_slopes_y_norm_no_nan, all_used_ones = init_data()
    if train_model_bool:
        train_model(model, all_WTD_no_nan, all_PmE_norm_no_nan, all_k_norm_no_nan, all_slopes_x_norm_no_nan, all_slopes_y_norm_no_nan, all_used_ones)
    if plot_MSE_dev_bool:
        plot_MSE_dev()
    if compute_overall_test_loss_bool:
        overall_test_loss, overall_test_loss_w_extreme = compute_overall_test_loss(model)
    if test_model_bool:
        test_model(model, overall_test_loss, overall_test_loss_w_extreme, save_estims_bool, which_test_samples_factor)
    if test_model_extreme_bool:
        test_model_extreme(model, overall_test_loss, overall_test_loss_w_extreme, save_extreme_estims_bool)
