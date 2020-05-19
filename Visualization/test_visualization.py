#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:00:54 2020

@author: vijetadeshpande
"""

import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Visualization')
from ChartClasses import FeatureSequencePlot
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Visualization')



# save plots for the predicted values
readdata = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/RNN results/0'
plotsavepath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/RNN results/0/plots'
lineplot_object = FeatureSequencePlot(datapath = readdata, 
                                      plot_title = 'Feature wise fit', 
                                      savepath = plotsavepath)

lineplot_object.save_results()


#
filepath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/0/OUTPUT FILES/input_tensor.json'