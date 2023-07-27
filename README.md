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

