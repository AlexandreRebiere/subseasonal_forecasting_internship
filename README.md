# subseasonal_forecasting_internship
This GitHub corresponds to the internship realized by Alexandre REBIERE from May to July 2023. It consisted in improving subseasonal forecasting by adding new features (Soil Moisture, Gross Primary Production, Ecosystem Respiration, Elevation, El Niño ...) to an existing dataset used in the RODEO forecast competition.

**State of the Art**

This GitHub aims at improving climate forecasts predicting precipitation or temperature anomalies starting from a dataset taken from a RODEO forecast competition to which we will add several relevant features. Please find informations relative to RODEO competition in this following paper : "https://arxiv.org/pdf/1809.07394.pdf" "Improving Subseasonal Forecasting in the Western U.S. with Machine Learning. Hwang, Oreinstein, Cohen, Pfeiffer, Mackey". The authors associated a Github repository : "https://github.com/paulo-o/forecast_rodeo" from which we used several functions to preprocess the data. The useful RODEO dataset can be found at "https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/DVN/IHBANG".

**Additional useful data**

During the study we decided to add some new relevant data such as Soil Moisture (CASM), Gross Primary Production (GPP) and Ecosystem Respiration (RECO) you can download below : 
- CASM : "https://zenodo.org/record/7072512#.ZEqF5ezMJAc"
- GPP/RECO : "https://zenodo.org/record/7761881" (only the monthly resolution had been added to our model)
- elevation can be found on this repository (new_features)
- El Nino correlations can be found at : "https://psl.noaa.gov/data/correlation/table.html"

**WARNING :**

This project has been made in a context of a very short (3 months) research internship at the LEAP center (Learning the Earth with AI & Physics), Columbia University, New York. It may contains mistakes or imprecisions. If you have any questions regarding the use or the results of this GitHub Repositery, please send an email to "alexandre.rebiere@espci.org" or directly to Pierre Gentine, the LEAP director, at "pg2328@columbia.edu". Please find more informations in the file "Internship_report:paper.pdf".

**First Step : Preprocess the data**

- In src/experiments open "create_data_matrices.ipynb". This notebook uses "experiments_util.py" functions, a file you can modify depending on the features you want to take into account. It creates and save a matrix containing time, latitude and longitude columns + all the features you want to study shifted with the amount of day you want depending on *target_horizon*.
- Once this previous matrice has been created, open "create_matrix_NN.ipynb". This notebook aims at reshaping the initial dataset, concatenating data with new features, create a pytorch tensor that will be used in a Pytorch Neural Network model.

**Second step : Train the model + evaluate performances**

Open the file "linear_model_predictions.ipynb" using functions from "tools.py". The model consists in a basic linear layer over the features using Pytorch. Performances are evaluated in terms of skills. Skills values can be stored in tensors of size (year, season, latitude, longitude), from year 2011 to 2017, so that you can study then skills improvement regarding seasons, year, or geographical areas. A similar study can be made on SHAP values.

**Third step : Analyzing the results**

Open the file "Visualization.ipynb" using functions from "Visualization_functions.py". In this notebook you will find different plots that reveals the climate forecast improvement when adding new_features. Many plots will help you analyze where (which climate region) predictions increases the most, when (seasons) and thanks to which features (SHAP values, Layerwise Relevance Propagation).
