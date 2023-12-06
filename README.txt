User's Manual

The Code is stored in the SRC directory. The paper and the presenation are stored in the DOC directory.

The Initial Findings of Correlation are found in the Correlation file. Make sure to import any packages that are stated at the top of each file. The corr.py file can be run using 'python corr.py' and The correlation file can be run using 'python correlation.py'. The results are printed in the terminal as outputs. 

The Models of ARIMA, XGBoost, and LSTM are found in the Models file. All the packages that need to be imported for the models are at the top of the files. Thile files can be run using 'python FILENMAE.py'. The resulting graphs will be saved as '.png' files to the directory after you run the python files. For reference, the resulting graphs are stored in the 'MODEL_plots' folders for each model respectively. The resulting RMSE scores for the models are returned in the terminal as output. The RMSE scores can be compared by knowing which one is lower and it usually the XGBoost one as described in our paper.
