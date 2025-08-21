Battery Lifetime Prediction Using Surface Temperature Features From Early Cycle Data
This repository contains code and workflows for early-cycle  battery lifetime prediction published at https://pubs.rsc.org/en/content/articlehtml/2025/ee/d4ee05179c. The project leverages multiple publicly available datasets and supports modular training and evaluation across different battery chemistries and cycling conditions.

Dataset Overview
We use several datasets in this work:
- TRI Dataset
From: K.A. Severson, P.M. Attia, et al.
Data-driven prediction of battery cycle life before capacity degradation, Nature Energy, 2019
Link: https://www.nature.com/articles/s41560-019-0356-8
- XJTU Dataset
From: F. Wang, Z. Zhai, Z. Zhao, Y. Di, X. Chen
Physics-informed neural network for lithium-ion battery degradation stable modeling and prognosis, Nature Communications, 2024
Link: https://www.nature.com/articles/s41467-024-48779-z
- Battery Archive (SNL and UL-PUR)
From: batteryarchive.org
- Y. Preger et al., Degradation of commercial lithium-ion cells as a function of chemistry and cycling conditions, J. Electrochem. Soc., 2020
Link: https://iopscience.iop.org/article/10.1149/1945-7111/abae37
- D. Juarez-Robles et al., Degradation-safety analytics in lithium-ion cells: Part I, J. Electrochem. Soc., 2020
Link: https://iopscience.iop.org/article/10.1149/1945-7111/abc8c0
(Note: Battery Archive datasets will be uploaded soon.)

Each dataset has its own folder containing:
- Data preprocessing scripts
- Model training routines
- Function utility scripts

See README.md for dataset download instructions in each dataset folder. Then, place the raw data inside the corresponding data folder.
Preprocess data
- For TRI: Run BuildPkl_BatchX.py (where X = 1, 2, or 3) to generate .pkl files.
- For XJTU: Run convertTo_npy.py to convert raw data into .npy format.
- Train models
Use the provided scripts to train and evaluate models using the preprocessed data.

Refer to requirements.txt in the main folder for version compatibility.

For inquiry of our work, please contact author yichunlu@cuhk.edu.hk for more information, or cite our work . Sugiarto, Z. Huang, Lu Y.-C. Lu, Battery Lifetime Prediction Using Surface Temperature Features from Early Cycle Data. Energy Environ. Sci., 2025, 18, 2511-2523.
