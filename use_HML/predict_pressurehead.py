#import os
import os
import sys
import yaml
import matplotlib.pyplot as plt
import numpy as np
from parflow.tools.io import read_pfb, write_pfb
import subsettools as st
import torch
import importlib


#########################################################################################################
########Convert estimated 2D DTWT to 3D pressure head by assuming hydrostatic equilibrium################
#########################################################################################################
def convert_WTD(estim, thickness_list=[200.0,100.0,50.0,25.0,10.0,5.0,1.0,0.6,0.3,0.1]):
    pressure_head = np.zeros((10, 150, 150))
    total_z = np.sum(thickness_list)
    for i in range(150):
        for j in range(150):
            sum_z = 0
            for k,z in enumerate(thickness_list):
                sum_z += z/2
                pressure_head[k,i,j] = (total_z - sum_z - estim[i,j])
                sum_z += z/2
    return pressure_head

#########################################################################################################
######Crop the four two-dimensional input features (k, pme, slope_x, slope_y) around desired basin#######
#########################################################################################################
def truncate_data(all_minmax_name, huc, thickness_list, gt_available_bool, plot_gt_bool):

    #Import input feature maps to be truncated on the desired basin
    slope_x = read_pfb('../data/slope_x.pfb')
    slope_y = read_pfb('../data/slope_y.pfb')
    k = read_pfb('../data/k.pfb')
    pme = read_pfb('../data/pme.pfb')
    all_minmax = torch.load('../train_HML/test_data/' + all_minmax_name).numpy()

    #ground truth if available, for comparison
    if gt_available_bool:
        WTD = np.load('../data/wtdepth3.npy')[2:-1, 1:-1]

    WTD_min = all_minmax[0][0]
    WTD_max = all_minmax[0][1]
    pme_min = all_minmax[1][0]
    pme_max = all_minmax[1][1]
    k_min = all_minmax[2][0]
    k_max = all_minmax[2][1]
    slope_x_min = all_minmax[3][0]
    slope_x_max = all_minmax[3][1]
    slope_y_min = all_minmax[4][0]
    slope_y_max = all_minmax[4][1] 

    # convert 3d k to 2d k_2d
    multi = 1
    for i in range(len(thickness_list)):
        multi += (thickness_list[i]/k[i,:,:])
    k_2d = 392.0 / multi 
    k_2d[k_2d==0] = np.nan

    ij_bounds, mask = st.define_huc_domain(hucs=[str(huc)], grid='conus2')
    print("ij_bounds returns [imin, jmin, imax, jmax]")
    print(f"bounding box on conus2: {ij_bounds}")
    nj = ij_bounds[3] - ij_bounds[1]
    ni = ij_bounds[2] - ij_bounds[0]

    shift_x = (150-ni)//2
    shift_y = (150-nj)//2
    trunc = [[ij_bounds[1]-shift_y, ij_bounds[1]-shift_y + 150], [ij_bounds[0]-shift_x, ij_bounds[0]-shift_x + 150]]

    k_trunc = k_2d[trunc[0][0]:trunc[0][1], trunc[1][0]:trunc[1][1]]
    pme_trunc = pme[-1,trunc[0][0]:trunc[0][1], trunc[1][0]:trunc[1][1]]
    slope_x_trunc = slope_x[0,trunc[0][0]:trunc[0][1], trunc[1][0]:trunc[1][1]]
    slope_y_trunc = slope_y[0,trunc[0][0]:trunc[0][1], trunc[1][0]:trunc[1][1]]

    k_trunc_norm = (k_trunc - k_min)/(k_max - k_min)
    pme_trunc_norm = (pme_trunc - pme_min)/(pme_max - pme_min)
    slope_x_trunc_norm = (slope_x_trunc - slope_x_min)/(slope_x_max - slope_x_min)
    slope_y_trunc_norm = (slope_y_trunc - slope_y_min)/(slope_y_max - slope_y_min)

    pme_trunc_norm = np.nan_to_num(pme_trunc_norm, nan=0.0)
    k_trunc_norm = np.nan_to_num(k_trunc_norm, nan=0.0)
    slope_x_trunc_norm = np.nan_to_num(slope_x_trunc_norm, nan=0.0)
    slope_y_trunc_norm = np.nan_to_num(slope_y_trunc_norm, nan=0.0)

    if gt_available_bool:
        WTD_trunc = WTD[trunc[0][0]:trunc[0][1], trunc[1][0]:trunc[1][1]]

        if plot_gt_bool:
            plt.figure()
            plt.imshow(WTD_trunc, cmap='twilight', origin='lower', vmin=WTD_min, vmax=WTD_max)
            plt.xlabel('Position [1000 m]', fontsize=12)
            plt.ylabel('Position [1000 m]', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.colorbar().set_label('DTWT [m]', fontsize=12)
            plt.savefig('plots/' + str(huc) + '_gt_DTWT_trunc.png', dpi=400)


    np_data = np.vstack(([k_trunc_norm], [pme_trunc_norm], [slope_x_trunc_norm], [slope_y_trunc_norm]))
    data_orig = torch.from_numpy(np_data).clone().detach().to(torch.float32)

    if gt_available_bool:
        label_orig = torch.tensor(WTD_trunc, dtype = torch.float32).unsqueeze(0)
    else:
        label_orig = torch.zeros((1, nj, ni), dtype=torch.float32)

    return data_orig, label_orig, trunc, WTD_min, WTD_max

#########################################################################################################
############################################Evaluate HML#################################################
#########################################################################################################
def evaluate_hml(model, data_orig, label_orig, huc, saving_nr, trunc, WTD_min, WTD_max, gt_available_bool, plot_pred_bool):

    ij_bounds, mask = st.define_huc_domain(hucs=[str(huc)], grid='conus2')

    model.eval()

    with torch.no_grad():
        res = model(data_orig)
        estim = res.float().cpu().detach().numpy()
        estim = estim.squeeze()
        if gt_available_bool:
            curr_loss = model.loss(res, label_orig).data.cpu().detach().numpy()
            label = label_orig.squeeze()

    pressure_head = convert_WTD(estim)
    pressure_head_trunc = pressure_head[:,ij_bounds[1]-trunc[0][0]:ij_bounds[3]-trunc[0][0], ij_bounds[0]-trunc[1][0]:ij_bounds[2]-trunc[1][0]]
    saving_path = 'generated_pressure_maps/' + saving_nr + '_' + str(huc) + '_pressure_head.pfb'
    write_pfb(saving_path, pressure_head_trunc)

    if plot_pred_bool:
        if gt_available_bool:
            fig, axes = plt.subplots(1,2, figsize=(20,10))
            axes[0].set_xlabel('Position [1000 m]', fontsize=20)
            axes[0].set_ylabel('Position [1000 m]', fontsize=20)
            axes[0].tick_params(axis='x', labelsize=18)
            axes[0].tick_params(axis='y', labelsize=18)
            im1 = axes[0].imshow(estim, cmap="twilight", origin='lower', vmin = WTD_min, vmax = WTD_max)#,norm=colors.LogNorm())
            axes[0].set_title("Modeled DTWT configuration, RMSE = "+str(np.sqrt(curr_loss))+".", fontsize = 24)

            im2 = axes[1].imshow(label, cmap="twilight", origin='lower', vmin = WTD_min, vmax = WTD_max)#,norm=colors.LogNorm())
            axes[1].set_title("Ground truth.", fontsize = 24)
            axes[1].set_xlabel('Position [1000 m]', fontsize=20)
            axes[1].set_ylabel('Position [1000 m]', fontsize=20)
            axes[1].tick_params(axis='x', labelsize=18)
            axes[1].tick_params(axis='y', labelsize=18)

            cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            fig.colorbar(im2, cax=cbar_ax, orientation='vertical').set_label('DTWT [m]', fontsize = 20)
            cbar_ax.tick_params(labelsize=18)

            imgsavingName = 'plots/' + str(huc) + '_estimation_vs_gt_DTWT.png'
            plt.savefig(imgsavingName, dpi=400)
            fig.show()
            plt.close()

        else:
            plt.figure()
            plt.xlabel('Position [1000 m]', fontsize=12)
            plt.ylabel('Position [1000 m]', fontsize=12)
            plt.xticks(fontsize=10)
            plt.yticks(fontsize=10)
            plt.imshow(estim, cmap="twilight", origin='lower', vmin = WTD_min, vmax = WTD_max)#,norm=colors.LogNorm())
            plt.colorbar().set_label('DTWT [m]', fontsize=12)
            plt.title("Predicted DTWT configuration.", fontsize = 12)

            # cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
            # fig.colorbar(im1, cax=cbar_ax, orientation='vertical').set_label('DTWT [m]', fontsize = 12)
            #cbar_ax.tick_params(labelsize=18)

            imgsavingName = 'plots/' + str(huc) + '_estimation_DTWT.png'
            plt.savefig(imgsavingName, dpi=400)
            plt.close()

if __name__ == "__main__":

    #########################################################################################################
    ####################################Import USER settings#################################################
    #########################################################################################################
    #config_file = '../train_HML/HML_settings.yaml'
    sys.path.append(os.path.abspath('../train_HML'))
    config_file = sys.argv[1]
    file = open(config_file)
    settings = yaml.load(file, Loader=yaml.FullLoader)

    saving_nr = settings['saving_nr']
    model_nr = settings['model_nr']
    ml_model = settings['ml_model']
    n_epochs = settings['n_epochs']
    seed = settings['seed']

    huc = settings['huc']
    gt_available_bool = settings['gt_available_bool']
    plot_gt_bool = settings['plot_gt_bool']
    plot_pred_bool = settings['plot_pred_bool']

    all_minmax_name = saving_nr + "_sd" + str(seed) + "_" + str(n_epochs) + "_epochs" + "_minmaxs.csv"

    #########################################################################################################
    ####################################Import trained model#################################################
    #########################################################################################################
    mldef = importlib.import_module(ml_model)
    model = mldef.my_model()
    model.eval()
    model_saving_name = model_nr + "_sd" + str(seed) + "_" + str(n_epochs) + "_epochs"
    mdlsavingName = '../train_HML/models/' + model_saving_name + '_my_trained_model.pth'
    model.load_state_dict(torch.load(mdlsavingName, map_location=torch.device('cpu')))

    thickness_list = [200.0,100.0,50.0,25.0,10.0,5.0,1.0,0.6,0.3,0.1]

    #########################################################################################################
    #######################Evaluate HML on desired basin, save estimation####################################
    #########################################################################################################
    data_orig, label_orig, trunc, WTD_min, WTD_max = truncate_data(all_minmax_name, huc, thickness_list, gt_available_bool, plot_gt_bool)
    evaluate_hml(model, data_orig, label_orig, huc, saving_nr, trunc, WTD_min, WTD_max, gt_available_bool, plot_pred_bool)