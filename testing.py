# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 22:53:24 2024

@author: Alfred
"""

import mne

file_dir = './BCICIV_2a_gdf/A01E.gdf'

ch_name = ['Fz',
           'FC3','FC1','FCz','FC2','FC4',
           'C5','C3','C1','Cz','C2','C4','C6',
           'CP3','CP1','CPz','CP2','CP4',
           'P1','Pz','P2',
           'POz',]

eog_name = ['EOG-left', 'EOG-central', 'EOG-right']

raw = mne.io.read_raw_gdf(file_dir, eog=eog_name, include=ch_name)

#%%
raw.plot(picks='eeg')