# HydroStartML: A combined machine learning and physics-based approach to reduce hydrological model spin-up time. 
Louisa Pawusch, Stefania Scheurer, Wolfgang Nowak, Reed Maxwell.
Preprint available on: https://doi.org/10.48550/arXiv.2504.17420

Abstract:
Finding the initial depth-to-water table (DTWT) configuration of a catchment is a critical challenge when simulating the hydrological cycle with integrated models, significantly impacting simulation outcomes. Traditionally, this involves iterative spin-up computations, where the model runs under constant atmospheric settings until steady-state is achieved. These so-called model spin-ups are computationally expensive, often requiring many years of simulated time, particularly when the initial DTWT configuration is far from steady state.
To accelerate the model spin-up process we developed HydroStartML, a machine learning emulator trained on steady-state DTWT configurations across the contiguous United States. HydroStartML predicts, based on available data like conductivity and surface slopes, a DTWT configuration of the respective watershed, which can be used as an initial DTWT.
Our results show that initializing spin-up computations with HydroStartML predictions leads to faster convergence than with other initial configurations like spatially constant DTWTs. The emulator accurately predicts configurations close to steady state, even for terrain configurations not seen in training, and allows especially significant reductions in computational spin-up effort in regions with deep DTWTs. This work opens the door for hybrid approaches that blend machine learning and traditional simulation, enhancing predictive accuracy and efficiency in hydrology for improving water resource management and understanding complex environmental interactions.

To execute, call the python files with the HML_settings.yaml file.
Find the data generation procedure in get_data/, the training of HydroStartML in train_HML/, the fully trained model in train_HML/models/, and examplary code on how to use the trained model on unseen terrain in use_HML/.
