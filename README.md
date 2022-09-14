# NKTCell-Interplay

Here we have uploaded the code to fit cancer cell data for a variety of experiments dealing with NK cell depletion.

The main code for fitting the data set and for generating plots of the fits can be found in NKTCellInterplay.py. Instructions for running the code are commented throughout the file. The data used in the code can be found in cellcounts.csv and subtype_proportions.csv.

After the original set of experiments were completed, a second set of experiments were performed where cytokines were depleted from the mice. In addition to fitting the original experiments, we took our fitted parameter values and applied them to the this second experiment to see how well our model could predict the results of the previous experiment. To simulate the loss of cytokines, a valuable resource in the tumor micro environment, we fit the parameter representing carrying capacity allowing it to change. The code for this prediction set can be found in cytokinedepletion.py. The data for the prediction code is found in cytokinedepletion.csv.
