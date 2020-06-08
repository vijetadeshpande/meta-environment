#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun May 17 17:00:54 2020

@author: vijetadeshpande
"""
import os
import sys
sys.path.insert(1, r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Visualization')
from ChartClasses import FeatureSequencePlot as PredictPlot
from ChartClasses import ErrorPlot as EPlot



# save plots for the predicted values
readdata = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/RNN results/3'
plotsavepath = os.path.join(readdata, 'plots3')
if not os.path.exists(plotsavepath):
    os.makedirs(plotsavepath)

# line plot for de-normalized predicted output (compared with CEPAC)
lineplot_object = PredictPlot(datapath = readdata, 
                              plot_title = 'Feature wise fit', 
                              savepath = plotsavepath)

lineplot_object.save_results()


# error plot
errorplot = EPlot(datapath = readdata,
                  plot_title = 'Error plot',
                  savepath = plotsavepath)
errorplot.plot_and_save()

#
filepath = r'/Users/vijetadeshpande/Documents/GitHub/meta-environment/Data and results/CEPAC RUNS/0/OUTPUT FILES/input_tensor.json'