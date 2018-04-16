# Met_data_predict
Attempt to use features from DAS dataset to predict meteorological data at fairbanks array 

This repo contains tests associated with this project. 

RFREG_test1.ipynb: first test of a random forest regressor on features calculated on 1 minute, nonoverlapping chunks of DAS data around probe M3_20
Results are not good, but this is unsurprising. Better fit to temperature than soil moisture. Could try grid search but unlikely to improve the fit significantly.

Issues
> Data may need better despiking  
> features calculated over too short of a window  
> maybe need different features  
> maybe need to train on filtered data  

