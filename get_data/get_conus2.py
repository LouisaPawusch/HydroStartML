import os
import sys
import yaml

import numpy as np
from parflow.tools.io import read_pfb, write_pfb
from parflow.tools.fs import mkdir
import subsettools as st
import hf_hydrodata as hf
# Defnet et al. (2024).  
# hf_hydrodata: A Python package for accessing hydrologic simulations and observations across the United States  
# Journal of Open Source Software, 9(99), 6623

# You need to register on https://hydrogen.princeton.edu/pin before you can use the hydrodata utilities
hf.register_api_pin("<your_email>", "<your_pin>")

config_file = sys.argv[1]
print("Using config file:", config_file)
file = open(config_file)
settings = yaml.load(file, Loader=yaml.FullLoader)

my_runname = settings['runname']
wy = settings['water_year']
grid = settings['grid']

start = "2005-10-01"
var_ds = "conus2_domain"
hucs = ["01","02","03","04","05","06","07","08","09","10","11","12","13","14","15","16","17","18"]

# cluster topology
P = 1
Q = 1

# set the directory paths where you want to write your subset files
input_dir = "../data/"

os.environ["PARFLOW_DIR"] = "path_to_your_parflow"

ij_bounds, mask = st.define_huc_domain(hucs=hucs, grid=grid)

nj = ij_bounds[3] - ij_bounds[1]
ni = ij_bounds[2] - ij_bounds[0]

mask_solid_paths = st.write_mask_solid(mask=mask, grid=grid, write_dir=input_dir)

static_paths = st.subset_static(ij_bounds, dataset=var_ds, write_dir=input_dir)

# WTD was then generated from pressurehead

pf_indicator = read_pfb(input_dir + 'pf_indicator.pfb')

k = np.zeros(pf_indicator.shape)
# Conductivities from Chen Yang, Danielle T. Tijerina-Kreuzer, Hoang V. Tran, Laura E. Condon, Reed M. Maxwell,
# A high-resolution, 3D groundwater-surface water simulation of the contiguous US: Advances in the integrated ParFlow CONUS 2.0 modeling platform,
# Journal of Hydrology,Volume 626, Part B, 2023, 130294, ISSN 0022-1694,
# https://doi.org/10.1016/j.jhydrol.2023.130294. (https://www.sciencedirect.com/science/article/pii/S0022169423012362)

k[pf_indicator==1] = 0.269022595
k[pf_indicator==2] = 0.043630356
k[pf_indicator==3] = 0.015841225
k[pf_indicator==4] = 0.007582087
k[pf_indicator==5] = 0.01818816
k[pf_indicator==6] = 0.005009435
k[pf_indicator==7] = 0.005492736
k[pf_indicator==8] = 0.004675077
k[pf_indicator==9] = 0.003386794
k[pf_indicator==10] = 0.004783973
k[pf_indicator==11] = 0.003979136
k[pf_indicator==12] = 0.006162952
k[pf_indicator==13] = 0.005009435

k[pf_indicator==19] = 0.005
k[pf_indicator==20] = 0.01
k[pf_indicator==21] = 0.02
k[pf_indicator==22] = 0.03
k[pf_indicator==23] = 0.04
k[pf_indicator==24] = 0.05
k[pf_indicator==25] = 0.06
k[pf_indicator==26] = 0.08
k[pf_indicator==27] = 0.1
k[pf_indicator==28] = 0.2

write_pfb(input_dir + 'k.pfb', k)