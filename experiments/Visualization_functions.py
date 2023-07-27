import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
import torch as th
import torch.nn as nn
from datetime import datetime, timedelta
import cartopy
import cartopy.crs as ccrs
import os

#------------------------------------------------------------------------------------------------------------------------------------------------------


def comparison_RODEO(gt_id, target_horizon, ground):
    """ Plots the training results and compare them to what has been observed in the initial RODEO paper."""
    if gt_id == "contest_tmp2m" : 
        if target_horizon == "56w":
            multillr = np.array([0.2522, 0.2313, 0.2212, 0.1585, 0.2694, 0.2213, 0.2562])
            autoknn = np.array([0.3240, 0.3205, 0.0531, 0.3056, 0.3939, 0.2882, 0.2817])
            ensemble = np.array([0.3537, 0.3193, 0.1833, 0.2643, 0.3752, 0.2933, 0.3025])
            rec_deb_cfs = np.array([0.3879, 0.1030, 0.1211, 0.1936, 0.4234, 0.0983, 0.1708])
            ens_cfs = np.array([0.4284, 0.3033, 0.1828, 0.3297, 0.4426, 0.2720, 0.3003])
            Title  = 'Skill from RODEO model for temperature anomalies forecasting over 5-6 weeks'
        if target_horizon == "34w":
            multillr = np.array([0.2695, 0.1466, 0.1031, 0.1973, 0.3513, 0.2654, 0.3079])
            autoknn = np.array([0.3664, 0.3135, 0.2011, 0.2775, 0.3885, 0.3502, 0.2807])
            ensemble = np.array([0.3525, 0.2548, 0.1852, 0.2935,  0.4269, 0.3467, 0.3451])
            rec_deb_cfs = np.array([0.4598, 0.1397, 0.2861, 0.3018, 0.2857, 0.2490, 0.0676])
            ens_cfs = np.array([0.4589, 0.2505, 0.2878, 0.3547, 0.4404, 0.3839, 0.3253])
            Title  = 'Skill from RODEO model for temperature anomalies forecasting over 3-4 weeks'
    if gt_id == "contest_precip" : 
        if target_horizon == "56w":
            multillr = np.array([0.1398, 0.3039, 0.1392, -0.0069, 0.0802, 0.1703, 0.1876])
            autoknn = np.array([0.2132, 0.3943, 0.1784, 0.0818, 0.0204, -0.0930, 0.1870])
            ensemble = np.array([0.2210, 0.4002, 0.2031, 0.0556, 0.0755, 0.0569, 0.2315])
            rec_deb_cfs = np.array([0.1835, 0.1941, 0.0782, 0.0155, 0.0292, -0.0160, -0.0038])
            ens_cfs = np.array([0.2666, 0.4224, 0.1939, 0.0782, 0.0959, 0.0483, 0.1978])
            Title  = 'Skill from RODEO model for precipitation anomalies forecasting over 5-6 weeks'
        if target_horizon == "34w":
            multillr = np.array([0.1817, 0.3147, 0.1552, 0.0790, 0.0645, 0.1419, 0.3079])
            autoknn = np.array([0.2173, 0.3648, 0.2026, 0.1208, -0.0053, -0.0568, 0.2807])
            ensemble = np.array([0.2420, 0.3983, 0.2130, 0.1391, 0.0532, 0.0636, 0.2364])
            rec_deb_cfs = np.array([0.1646, 0.0828, 0.0648, 0.1272, 0.0837, 0.0190, 0.0596])
            ens_cfs = np.array([0.2692, 0.3909, 0.1711, 0.1738, 0.1043, 0.0435, 0.2250])
            Title  = 'Skill from RODEO model for precipitation anomalies forecasting over 3-4 weeks'
            
    mean_skill_store = [np.mean(multillr), np.mean(autoknn), np.mean(ensemble), np.mean(rec_deb_cfs), np.mean(ens_cfs), np.mean(ground)]
    methods = ['multillr', 'autoknn', 'ensemble', 'rec_deb_cfs', 'ens_cfs', 'result']
    df = pd.DataFrame({'methods':methods, 'mean_skill':mean_skill_store})
    print(df)
    years = np.array([2011+i for i in range(7)])
    weights = np.arange(1, 5)
    plt.figure(figsize=(10, 8))
    plt.scatter(years, multillr, cmap='Greys', marker='+')
    plt.plot(years,multillr, '-o', label="multillr", c='0.05')
    plt.plot(years,autoknn,'r-o', label="autoknn", c='0.25')
    plt.plot(years,ensemble,'b-o', label="ensemble",c='0.45')
    plt.plot(years,rec_deb_cfs,'c-o', label="rec_deb_cfs",c='0.65')
    plt.plot(years,ens_cfs,'y-o', label="ens_cfs",c='0.85')
    plt.plot(years,ground,'r-*', label="initial features", markersize=20)
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Skill')
    plt.title(Title) 
    plt.show()
    

#------------------------------------------------------------------------------------------------------------------------------------------------------    


def plot_skill_simulation(skills, year, mask):
    """ Shows skills geaographically, season by season, averaged over all years from 2011 to 2017"""
    years=[2011+i for i in range(7)]
    seasons = ['winter','spring','summer','fall']
    year_index = years.index(year)
    data = skills  
    # Define the map boundaries
    llcrnrlon = -124
    llcrnrlat = 27
    urcrnrlon = -94
    urcrnrlat = 49
    fig, ax = plt.subplots(1, 4, figsize=(25,25))
    for i, season in enumerate(seasons):
        m = Basemap(projection='cyl', llcrnrlon=-125, llcrnrlat=26, urcrnrlon=-93, urcrnrlat=50, resolution='c', ax=ax[i])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        # Convert the latitude and longitude to map coordinates
        lons, lats = np.meshgrid(np.arange(llcrnrlon, urcrnrlon+1,1), np.arange(llcrnrlat, urcrnrlat+1, 1))
        x, y = m(lons, lats)
    
        masked_data = np.ma.masked_array(data[year_index,i].flatten(), mask=mask.flatten())
        # Scatter plot with colored points based on data values
        sc = ax[i].scatter(x, y, c=masked_data, cmap='jet', s=50, vmin=-1, vmax=1)  # Adjust 'cmap' and 's' as desired
        ax[i].set_title('average skills for year ' + str(year) +' '+ season)
        cbar = plt.colorbar(sc, ax=ax[i], shrink=0.08)
    plt.show()
    

#------------------------------------------------------------------------------------------------------------------------------------------------------  


def differences_skill_simulation(skills1, skills2, year, feature, mask):
    """ Plots on 4 maps (one for each season) the difference in skills betwee two trainings with different datasets. It highlights the zones where we observe improvements or degradation"""
    years=[2011+i for i in range(7)]
    llcrnrlon = -124
    llcrnrlat = 27
    urcrnrlon = -94
    urcrnrlat = 49
    year_index = years.index(year)
    seasons = ['winter','spring','summer','fall']
    
    fig, ax = plt.subplots(1, 4, figsize=(25,25))
    for i, season in enumerate(seasons):
        data = skills1[year_index, i,:,:]-skills2[year_index,i,:,:]
        m = Basemap(projection='cyl', llcrnrlon=-125, llcrnrlat=26, urcrnrlon=-93, urcrnrlat=50, resolution='c', ax=ax[i])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        # Convert the latitude and longitude to map coordinates
        lons, lats = np.meshgrid(np.arange(llcrnrlon, urcrnrlon+1,1), np.arange(llcrnrlat, urcrnrlat+1, 1))
        x, y = m(lons, lats)
    
        masked_data = np.ma.masked_array(data.flatten(), mask=mask.flatten())
        # Scatter plot with colored points based on data values
        sc = ax[i].scatter(x, y, c=masked_data, cmap='jet', s=50, vmin=-0.5, vmax=0.5)  # Adjust 'cmap' and 's' as desired
        ax[i].set_title(season)
        cbar = plt.colorbar(sc, ax=ax[i], shrink=0.07)
    fig.suptitle('skill improvement in '+ str(year) + ' using ' + feature, fontsize=30, y=0.6)
    fig.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------  


def season_mean(skills1, skills2, feature, mask): 
    """ Averages skills over thes seasons for all year between 2011-2017, returns 4 maps corresponding to each season."""
    years=[2011+i for i in range(7)]
    llcrnrlon = -124
    llcrnrlat = 27
    urcrnrlon = -94
    urcrnrlat = 49
    seasons = ['winter','spring','summer','fall']
    fig, ax = plt.subplots(1, 4, figsize=(25,25))
    data=th.zeros(len(years),len(seasons),23,31)
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            data_row = skills1[j, i,:,:]-skills2[j,i,:,:]
            data[j,i,:,:] = data_row
        data_mean = data.mean(axis=0)
        m = Basemap(projection='cyl', llcrnrlon=-125, llcrnrlat=26, urcrnrlon=-93, urcrnrlat=50, resolution='c', ax=ax[i])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        # Convert the latitude and longitude to map coordinates
        lons, lats = np.meshgrid(np.arange(llcrnrlon, urcrnrlon+1,1), np.arange(llcrnrlat, urcrnrlat+1, 1))
        x, y = m(lons, lats)
        masked_data = np.ma.masked_array(data_mean[i,:,:].flatten(), mask=mask.flatten())
        # Scatter plot with colored points based on data values
        sc = ax[i].scatter(x, y, c=masked_data, cmap='jet', s=50)#, vmin=-0.50, vmax=0.50)  # Adjust 'cmap' and 's' as desired
        ax[i].set_title(season)
        cbar = plt.colorbar(sc, ax=ax[i], shrink=0.07)
    fig.suptitle('skill improvement per season averaged in 2011-2017 using ' + feature, fontsize=30, y=0.6)
    fig.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------


def season_mean_except_2013(skills1, skills2, feature, mask): 
    """ Averages skills over thes seasons for all year between 2011-2017 except 2013, returns 4 maps corresponding to each season. Same as season_mea, but without 2013."""
    years=[2011+i for i in range(7)]
    llcrnrlon = -124
    llcrnrlat = 27
    urcrnrlon = -94
    urcrnrlat = 49
    seasons = ['winter','spring','summer','fall']
    fig, ax = plt.subplots(1, 4, figsize=(25,25))
    data=th.zeros(len(years)-1,len(seasons),23,31)
    for i, season in enumerate(seasons):
        for j, year in enumerate(years):
            if year !=2013 : 
                data_row = skills1[j, i,:,:]-skills2[j,i,:,:]
                if year<2013 : 
                    data[j,i,:,:] = data_row
                elif year>2013 : 
                    data[j-1,i,:,:] = data_row
        data_mean = data.mean(axis=0)
        m = Basemap(projection='cyl', llcrnrlon=-125, llcrnrlat=26, urcrnrlon=-93, urcrnrlat=50, resolution='c', ax=ax[i])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        # Convert the latitude and longitude to map coordinates
        lons, lats = np.meshgrid(np.arange(llcrnrlon, urcrnrlon+1,1), np.arange(llcrnrlat, urcrnrlat+1, 1))
        x, y = m(lons, lats)
        masked_data = np.ma.masked_array(data_mean[i,:,:].flatten(), mask=mask.flatten())
        # Scatter plot with colored points based on data values
        sc = ax[i].scatter(x, y, c=masked_data, cmap='jet', s=50)#, vmin=-0.50, vmax=0.50)  # Adjust 'cmap' and 's' as desired
        ax[i].set_title(season)
        cbar = plt.colorbar(sc, ax=ax[i], shrink=0.07)
    fig.suptitle('skill improvement per season averaged in 2011-2017 (except 2013) using ' + feature, fontsize=30, y=0.6)
    fig.tight_layout()
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------  
    
    
def subplot_skills(skills1, skills2, feature, mask, gt_id, target_horizon):
    """ Compares the skills resultig from the model using two different datasets """
    masked_data1 = np.ma.masked_array(skills1, mask=mask.repeat(7,4,1,1))
    masked_data2 = np.ma.masked_array(skills2, mask=mask.repeat(7,4,1,1))
    mean1 = masked_data1.mean(axis=(1,2,3))
    mean2 = masked_data2.mean(axis=(1,2,3))
    years=[2011+i for i in range(7)]
    plt.plot(years,mean1,'r-o', label='initial data')
    plt.plot(years,mean2,'b-+', label='initial data + ' + feature )
    plt.legend()
    plt.xlabel('Year')
    plt.xticks(np.arange(min(years), max(years)+1, 1.0))
    plt.ylabel('Skill')
    if gt_id == "contest_tmp2m":
        if target_horizon == "56w":
            plt.title('Skill comparison over test_set for temperature anomalies forecasting over 5-6 weeks')
        if target_horizon == "34w":
            plt.title('Skill comparison over test_set for temperature anomalies forecasting over 3-4 weeks')
    if gt_id == "contest_precip":
        if target_horizon == "56w":
            plt.title('Skill comparison over test_set for precipitation anomalies forecasting over 5-6 weeks')
        if target_horizon == "34w":
            plt.title('Skill comparison over test_set for precipitation anomalies forecasting over 3-4 weeks')
    plt.show()

    
#------------------------------------------------------------------------------------------------------------------------------------------------------ 


def link_improvement_feature(skills1, skills2, gt_id, target_horizon, feature, index_feature, mask):
    """ Plots the scatterplot of skill improvements values depending on the feature value. A fittig curve is also applied to show the behaviour of the scatterplot."""
    ######load the considered timelist
    extract_time = th.load('results/matrix/time.tensor')
    extract_time = np.array(extract_time).tolist()
    datetime_objects = []
    for timestamp in extract_time:
        timestamp_seconds = timestamp / 1e9
        dt = datetime.fromtimestamp(timestamp_seconds)
        datetime_objects.append(dt)
    datetime_objects = np.array(datetime_objects)

    #######load the data
    skill_improvement = (skills2 / skills1 - 1 ) * 100

    if gt_id == "contest_tmp2m":
        if target_horizon == "56w":
            ground_data = th.load('results/matrix/data_tmp2m56_new.tensor')
            Title = 'skill improvement for temperature anomalies 5-6 depending on ' + feature + ' data'
        if target_horizon == "34w":
            ground_data = th.load('results/matrix/data_tmp2m34_new.tensor')
            Title = 'skill improvement for temperature anomalies 3-4 depending on ' + feature + ' data'
    if gt_id == "contest_precip":
        if target_horizon == "56w":
            ground_data = th.load('results/matrix/data_precip56_new.tensor') 
            Title = 'skill improvement for precipitation anomalies 5-6 depending on ' + feature + ' data'
        if target_horizon == "34w":
            ground_data = th.load('results/matrix/data_precip34_new.tensor')
            Title = 'skill improvement for precipitation anomalies 3-4 depending on ' + feature + ' data'
    feature_data = ground_data[:,:,:,index_feature] 
    
    i=0
    while feature_data[i,:,:].max() == 0:
        i+=1
    for day in range(i+1,feature_data.shape[0]):
        if feature_data[day,:,:].max() == 0:
            feature_data[day,:,:] = feature_data[day-1,:,:] 
        

    ########manipulate the data

    years = [2011+i for i in range(7)]
    seasons = ['winter', 'spring', 'summer', 'fall']
    SM_data = th.zeros((len(years),len(seasons),mask.shape[0],mask.shape[1]))
    skill_full_list=[]
    for k,year in enumerate(years):
        start_date = [dt for dt in datetime_objects if dt.year == year]
        skill_list = []
        for indexing, season in enumerate(seasons):
            if indexing==0 : #winter
                date_season = start_date[:80]
                skill_list.append(skill_improvement[k, indexing, :, :].unsqueeze(0))
            if indexing==1 : #spring
                date_season = start_date[80:172]
                skill_list.append(skill_improvement[k, indexing, :, :].unsqueeze(0))
            if indexing==2 : #summer
                date_season = start_date[172:265]
                skill_list.append(skill_improvement[k, indexing, :, :].unsqueeze(0))
            if indexing==3 : #fall
                date_season = start_date[265:-9]
                skill_list.append(skill_improvement[k, indexing, :, :].unsqueeze(0))
            indexes_year = [i for i, dt in enumerate(datetime_objects) if dt in date_season] #extract the indexes of the year of interest
            SM = feature_data[indexes_year,:,:]
            SM_data[k,indexing,:,:] = SM.mean(axis=0)
        skill_list2 = th.concatenate(skill_list).unsqueeze(0)
        skill_full_list.append(skill_list2)
    skill_full_list = th.concatenate(skill_full_list)

    ### flattening the tensors
    SM_flattened = SM_data.flatten()
    skills_improvement_flatten = skill_full_list.flatten()

    ###plotting conditions
    condition_indices = th.nonzero((skills_improvement_flatten > -100) & (skills_improvement_flatten<100)).squeeze()
    condition_skill = skills_improvement_flatten[condition_indices].numpy()
    condition_SM = SM_flattened[condition_indices].numpy()

    #polynomial curve
    degree = 2
    coefficients = np.polyfit(condition_SM, condition_skill, degree)
    curve_fit = np.poly1d(coefficients)
    x_curve = np.linspace(0, condition_SM.max(), 100)
    y_curve = curve_fit(x_curve)

    ####Plot
    plt.plot(x_curve, y_curve, color='red', label='Fitted Curve')
    plt.scatter(condition_SM,condition_skill, marker = '+', label="skill improvement depending on " + feature + " data")
    plt.legend()
    plt.xlabel('Soil Moisture')
    plt.ylabel('Skill improvement 2011-2017')
    plt.title(Title)
    plt.show()
    
    
#------------------------------------------------------------------------------------------------------------------------------------------------------ 


def transform_large_region(row):
    """ Separates mild climate zones betwee oceanic and continental. """ 
    if row['Large region']=='C':
        if row['lon'] < 240 or (row['lon'] >= 250 and row['lat'] < 31):
            return 'Co'
        else:
            return 'Cc'
    else:
        return row['Large region']

    
#------------------------------------------------------------------------------------------------------------------------------------------------------


def climate_zone_preprocess(gt_id, target_horizon):
    """ Restriction from the worldwide Koppenn classification to the interesting zone used in the study. """
    file_path = "data/clusters coordinates/koppen_1901-2010.tsv"
    df = pd.read_csv(file_path, delimiter='\t')
    
    if gt_id == "contest_tmp2m":
        if target_horizon == "56w" :
            data=pd.read_hdf('results/regression/shared/contest_tmp2m_56w/lat_lon_date_data-contest_tmp2m_56wnew.h5')
        if target_horizon =="34w" :
            data=pd.read_hdf('results/regression/shared/contest_tmp2m_34w/lat_lon_date_data-contest_tmp2m_34wnew.h5')
    if gt_id == "contest_precip":
        if target_horizon == "56w" :
            data=pd.read_hdf('results/regression/shared/contest_precip_56w/lat_lon_date_data-contest_precip_56wnew.h5')
        if target_horizon == "34w" :
            data=pd.read_hdf('results/regression/shared/contest_precip_34w/lat_lon_date_data-contest_precip_34wnew.h5')
    
    
    df_climate_zones=df[(df['longitude']>=-125)&(df['longitude']<=-93)&(df['latitude']>=26)&(df['latitude']<=50)]
    df_climate_zones.reset_index(inplace=True,drop=True)
    df_climate_zones['longitude']=df_climate_zones['longitude']+360
    df_climate_zones = df_climate_zones.sort_values(by=['latitude', 'longitude'])
    df_climate_zones['latitude'] = df_climate_zones['latitude'].astype(float)
    df_climate_zones['longitude'] = df_climate_zones['longitude'].astype(float)
    df_climate_zones['Large region'] = df_climate_zones['p1901_2010'].apply(lambda x: x[0])
    df_climate_zones['latitude'] = df_climate_zones['latitude'].apply(lambda x: round(x) if round(x % 1, 2) == 0.25 else x)
    df_climate_zones['longitude'] = df_climate_zones['longitude'].apply(lambda x: round(x) if round(x % 1, 2) == 0.25 else x)
    df_climate_zones.rename(columns={'latitude':'lat','longitude':'lon'},inplace=True)
    data=data.sort_values(by=['start_date'])
    coordinates=data[['lat','lon']][0:514]
    coordinates=coordinates.sort_values(by=['lat','lon'])
    coordinates.reset_index(inplace=True,drop=True)
    coordinates=pd.merge(coordinates,df_climate_zones,on=['lat','lon'],how='left')
    coordinates['Large region'] = coordinates.apply(transform_large_region, axis=1)
    return(coordinates)
    

#------------------------------------------------------------------------------------------------------------------------------------------------------

    
def climate_zone_maps(coordinates):
    """ Creates the matrix separating i climate zones """
    map = ccrs.LambertConformal(central_longitude=-95, central_latitude=37.5, standard_parallels=(33, 45))
    plt.figure(figsize=(12, 8))
    ax = plt.axes(projection=map)
    ax.set_extent([-130, -65, 23, 50])
    ax.coastlines(resolution='10m', color='black', linewidth=0.5)
    ax.add_feature(cartopy.feature.LAND, facecolor='white', edgecolor='black')
    ax.add_feature(cartopy.feature.OCEAN, facecolor='lightblue')
    zone_colors = {
        'B': 'red',
        'Cc':'blue',
        'Co': 'cyan',
        'D': 'green',
        'E': 'orange'
    }

    for index, row in coordinates.iterrows():
        lat = row['lat']
        lon = row['lon']
        climate_zone = row['Large region']
        ax.plot(lon, lat, marker='o', color=zone_colors.get(climate_zone, 'black'), markersize=5, transform=ccrs.PlateCarree())
    legend_labels = ['B - Dry climates', 'Cc - Mild temperate continental regions', 'Co - Mild temperate oceanic regions','D Continental', 'E - Polar']
    legend_colors = ['red', 'blue', 'cyan','green', 'orange']
    legend_elements = [plt.Line2D([0], [0], marker='o', color=color, label=label, linestyle='') for label, color in zip(legend_labels, legend_colors)]
    ax.legend(handles=legend_elements, loc='lower right')
    plt.show()
    
    
#------------------------------------------------------------------------------------------------------------------------------------------------------
    
    
def save_koppen(coordinates):
    """ Saves the Koppen classificatio for climate zones in tensor"""
    extract_latitudes = coordinates['lat'].drop_duplicates().sort_values()
    extract_longitudes = coordinates['lon'].drop_duplicates().sort_values()
    extract_latitudes =np.array(extract_latitudes).tolist()
    extract_longitudes =np.array(extract_longitudes).tolist()
    # Create the coordinate map

    coordinates_float = coordinates.replace(['B','Cc','Co', 'D', 'E'],[1,2,3,4,5])
    print(coordinates_float)

    coordinate_map = {}
    for lat_idx, lat in enumerate(extract_latitudes):
        for lon_idx, lon in enumerate(extract_longitudes):
            coordinate_map[(lat, lon)] = (lat_idx, lon_idx)

    # Create an empty tensor with the desired dimensions
    tensor_Koppen = th.empty((len(extract_latitudes), len(extract_longitudes)))

    # Iterate over the dataframe rows and fill the tensor using the coordinate map
    for _, row in coordinates_float.iterrows():
        lat, lon, feature = row['lat'], row['lon'], row['Large region']
        tensor_Koppen[coordinate_map[(lat, lon)]] = feature
    # Print the filled tensor
    print(tensor_Koppen.shape)

    cache_dir = os.path.join('results', 'matrix')
    # e.g., cache_dir = 'results/regression/shared/contest_precip_34w'

    # if cache_dir doesn't exist, create it
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    # Filenames for data file to be stored in cache_dir
    data_file = os.path.join(
    cache_dir, "Koppen.tensor")

    print("Saving multiarrays features to " + data_file)
    th.save(tensor_Koppen, data_file)

    print("Finished generating data matrix.")
    

#------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_koppen(koppen_tensor, mask):
    """ Plots the climate region ovr the interesting region"""
    data = koppen_tensor  
    # Define the map boundaries
    llcrnrlon = -124
    llcrnrlat = 27
    urcrnrlon = -94
    urcrnrlat = 49
    
    # Create the figure and axis
    fig, ax = plt.subplots(1, 1, figsize=(15,15))
    m = Basemap(projection='cyl', llcrnrlon=-125, llcrnrlat=26, urcrnrlon=-93, urcrnrlat=50, resolution='c', ax=ax)
    m.drawcoastlines()
    m.drawcountries()
    m.drawstates()
    # Convert the latitude and longitude to map coordinates
    lons, lats = np.meshgrid(np.arange(llcrnrlon, urcrnrlon+1,1), np.arange(llcrnrlat, urcrnrlat+1, 1))
    x, y = m(lons, lats)
    
    masked_data = np.ma.masked_array(data.flatten(), mask=mask.flatten())
    #print(masked_data.shape)
    # Scatter plot with colored points based on data values
    sc = ax.scatter(x, y, c=masked_data, cmap='jet', s=600)#, vmin=-0.1, vmax=0.7)  # Adjust 'cmap' and 's' as desired
    ax.set_title('Koppen climate regions')
    
   # Define the legend labels
    legend_labels = {1: 'B - Dry climates', 2: 'Cc - Mild temperate continental regions', 3: 'Co - Mild temperate oceanic regions', 4: 'D Continental', 5: 'E - Polar'}

    # Create a custom legend with the defined labels and matching colors
    legend_handles = []
    for label_value, label_text in legend_labels.items():
        color = sc.to_rgba(label_value)  # Get the color from the scatter plot's colormap
        legend_handles.append(plt.Line2D([], [], marker='o', color='w', markerfacecolor=color, markersize=10, label=f'{label_text}'))

    # Add the legend to the plot
    ax.legend(handles=legend_handles, title='Legend')
    plt.show()

#------------------------------------------------------------------------------------------------------------------------------------------------------

def climate_histograms_skills(skills1, skills2, koppen_tensor, feature, criteria):
    """ Compute histograms showing the proportion of points per climate region where the skills have been improved by more than X % """
    climate_zones = ['B - Dry climates', 'Cc - Mild continental', 'Co - Mild oceanic','D Continental', 'E - Polar']
    skills_difference = skills1 - skills2
    skills_difference = skills_difference.reshape(skills_difference.shape[0]*skills_difference.shape[1],skills_difference.shape[2], skills_difference.shape[3])
    count_skill_improvement = np.zeros((5,1))
    count_list_tot=np.zeros((5,1))
    for latitude in range(skills1.shape[2]):
        for longitude in range(skills1.shape[3]):
            if koppen_tensor[latitude,longitude] == 1:
                count_list_tot[0]+=1
            elif koppen_tensor[latitude,longitude] == 2:
                count_list_tot[1]+=1
            elif koppen_tensor[latitude,longitude] == 3:
                count_list_tot[2]+=1
            elif koppen_tensor[latitude,longitude] == 4:
                count_list_tot[3]+=1
            elif koppen_tensor[latitude,longitude] == 5:
                count_list_tot[4]+=1
            for date in range(skills_difference.shape[0]):
                if skills_difference[date, latitude, longitude]>criteria:
                    if koppen_tensor[latitude,longitude] == 1:
                        count_skill_improvement[0]+=1
                    elif koppen_tensor[latitude,longitude] == 2:
                        count_skill_improvement[1]+=1
                    elif koppen_tensor[latitude,longitude] == 3:
                        count_skill_improvement[2]+=1
                    elif koppen_tensor[latitude,longitude] == 4:
                        count_skill_improvement[3]+=1
                    elif koppen_tensor[latitude,longitude] == 5:
                        count_skill_improvement[4]+=1                    
    count_skill_improvement = count_skill_improvement.T[0]
    count_list_tot = count_list_tot.T[0]*skills_difference.shape[0]
    percentage_count = (count_skill_improvement/count_list_tot)*100

    bin_width = 0.8
    x = np.arange(len(climate_zones)) + 0.5 * bin_width
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(x, percentage_count, width=bin_width, edgecolor='black', color='skyblue')
    ax.set_xticks(x)
    ax.set_xticklabels(climate_zones)
    ax.set_ylabel('Frequency (%)')
    ax.set_title('Histogram representing the % of points in climate regions where the skills have been improved by more than ' + str(criteria*100) + ' % with the addition of ' + feature)
    plt.show()
    

#------------------------------------------------------------------------------------------------------------------------------------------------------


def climate_histograms_SHAP(shap, gt_id, feature, koppen_tensor, criteria):
    """ Compute histograms showing the proportion of SHAP values > x (if x>0) or < x (if x<0), per climate region"""
    climate_zones = ['B - Dry climates', 'Cc - Mild continental', 'Co - Mild oceanic','D Continental', 'E - Polar']
    if gt_id == 'contest_tmp2m':
        feature_names_temp = ['rhum_shift44', 'pres_shift44','nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
           'tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86',
           'tmp2m_shift86_anom', 'mei_shift59', 'phase_shift31',
           'sst_2010_1_shift44', 'sst_2010_2_shift44', 'sst_2010_3_shift44',
           'icec_2010_1_shift44', 'icec_2010_2_shift44', 'icec_2010_3_shift44',
           'wind_hgt_10_2010_1_shift44', 'wind_hgt_10_2010_2_shift44', 'CASM', 'GPP', 'RECO', 'elevation', 'ElNino1', 'ElNino34', 'ElNino4' ]
        feature_of_interest  = feature_names_temp.index(feature)
        shap_difference = shap[:,:,:,feature_of_interest]
    elif gt_id == 'contest_precip':
        feature_names_precip =  ['rhum_shift44','pres_shift44', 'nmme_wo_ccsm3_nasa','nmme0_wo_ccsm3_nasa',
            'precip_shift43','precip_shift43_anom',
            'tmp2m_shift43','tmp2m_shift43_anom','precip_shift86','precip_shift86_anom','tmp2m_shift86',
            'tmp2m_shift86_anom','mei_shift59','phase_shift31','sst_2010_1_shift44','sst_2010_2_shift44',
            'sst_2010_3_shift44','icec_2010_1_shift44','icec_2010_2_shift44','icec_2010_3_shift44','wind_hgt_10_2010_1_shift44',
            'wind_hgt_10_2010_2_shift44','CASM','GPP','RECO','elevation', 'ElNino1', 'ElNino34', 'ElNino4']
        feature_of_interest  = feature_names_precip.index(feature)
        shap_difference = shap[:,:,:,feature_of_interest]
    count_shap_high = np.zeros((5,1))
    count_list_tot=np.zeros((5,1))
    for latitude in range(shap_difference.shape[1]):
        for longitude in range(shap_difference.shape[2]):
            if koppen_tensor[latitude,longitude] == 1:
                count_list_tot[0]+=1
            elif koppen_tensor[latitude,longitude] == 2:
                count_list_tot[1]+=1
            elif koppen_tensor[latitude,longitude] == 3:
                count_list_tot[2]+=1
            elif koppen_tensor[latitude,longitude] == 4:
                count_list_tot[3]+=1
            elif koppen_tensor[latitude,longitude] == 5:
                count_list_tot[4]+=1
            for date in range(shap_difference.shape[0]):
                if criteria >0 and shap_difference[date, latitude, longitude]>criteria:
                    if koppen_tensor[latitude,longitude] == 1:
                        count_shap_high[0]+=1
                    elif koppen_tensor[latitude,longitude] == 2:
                        count_shap_high[1]+=1
                    elif koppen_tensor[latitude,longitude] == 3:
                        count_shap_high[2]+=1
                    elif koppen_tensor[latitude,longitude] == 4:
                        count_shap_high[3]+=1
                    elif koppen_tensor[latitude,longitude] == 5:
                        count_shap_high[4]+=1
                elif criteria <=0 and shap_difference[date, latitude, longitude]<criteria:
                    if koppen_tensor[latitude,longitude] == 1:
                        count_shap_high[0]+=1
                    elif koppen_tensor[latitude,longitude] == 2:
                        count_shap_high[1]+=1
                    elif koppen_tensor[latitude,longitude] == 3:
                        count_shap_high[2]+=1
                    elif koppen_tensor[latitude,longitude] == 4:
                        count_shap_high[3]+=1
                    elif koppen_tensor[latitude,longitude] == 5:
                        count_shap_high[4]+=1
    count_shap_high = count_shap_high.T[0]
    count_list_tot = count_list_tot.T[0]*shap_difference.shape[0]
    percentage_count = (count_shap_high/count_list_tot)*100

    bin_width = 0.8
    x = np.arange(len(climate_zones)) + 0.5 * bin_width
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.bar(x, percentage_count, width=bin_width, edgecolor='black', color='purple')
    ax.set_xticks(x)
    ax.set_xticklabels(climate_zones)
    ax.set_ylabel('Frequency (%)')
    if criteria>0:
        ax.set_title('Histogram representing the % of points in climate regions where the shap value is higher than ' + str(criteria) + ' conceirning ' + feature)
    elif criteria <=0:
        ax.set_title('Histogram representing the % of points in climate regions where the shap value is lower than ' + str(criteria) + ' conceirning ' + feature)
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------


def seasonal_SHAP_value(shap, gt_id):
    """ Plots the role of the added features, looking their average SHAP value per season, compared to all the featurs that are not geographically constant indexes."""
    if gt_id == 'contest_tmp2m':
        feature_names = ['rhum_shift44', 'pres_shift44','nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
           'tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86',
           'tmp2m_shift86_anom', 'mei_shift59', 'phase_shift31',
           'sst_2010_1_shift44', 'sst_2010_2_shift44', 'sst_2010_3_shift44',
           'icec_2010_1_shift44', 'icec_2010_2_shift44', 'icec_2010_3_shift44',
           'wind_hgt_10_2010_1_shift44', 'wind_hgt_10_2010_2_shift44', 'CASM', 'GPP', 'RECO', 'elevation', 'ElNino1', 'ElNino34', 'ElNino4' ]
        SHAP_reduced = np.concatenate((shap[:,:,:,0:8],shap[:,:,:,18:22]), axis=3) #we eliminate features used as idexes to gain clarity
        features_names_useful = feature_names[0:8] + feature_names[18:22]

    elif gt_id == 'contest_precip':
        feature_names =  ['rhum_shift44','pres_shift44', 'nmme_wo_ccsm3_nasa','nmme0_wo_ccsm3_nasa',
            'precip_shift43','precip_shift43_anom',
            'tmp2m_shift43','tmp2m_shift43_anom','precip_shift86','precip_shift86_anom','tmp2m_shift86',
            'tmp2m_shift86_anom','mei_shift59','phase_shift31','sst_2010_1_shift44','sst_2010_2_shift44',
            'sst_2010_3_shift44','icec_2010_1_shift44','icec_2010_2_shift44','icec_2010_3_shift44','wind_hgt_10_2010_1_shift44',
            'wind_hgt_10_2010_2_shift44','CASM','GPP','RECO','elevation', 'ElNino1', 'ElNino34', 'ElNino4']
        SHAP_reduced = np.concatenate((shap[:,:,:,0:12],shap[:,:,:,22:26]), axis=3) #we eliminate features used as idexes to gain clarity
        features_names_useful = feature_names[0:12] + feature_names[22:26]
    seasons = ['winter', 'spring', 'summer', 'fall']
    fig, ax = plt.subplots(nrows=1, ncols=4,figsize=(20, 10), sharex =  True)
    for indexing, season in enumerate(seasons):
        if indexing == 0:
            shapi = SHAP_reduced[:80]
        if indexing == 1:
            shapi = SHAP_reduced[80:172]
        if indexing == 2:
            shapi = SHAP_reduced[172:265]
        if indexing == 3:
            shapi = SHAP_reduced[265:-9]
        average_shap_values = abs(shapi.mean(axis=(0,1,2)))
        idx = np.argsort(average_shap_values)
        indexes_sorted = [features_names_useful[i] for i in idx.tolist()]
        average_shap_values = [average_shap_values[i] for i in idx.tolist()]
        Y_axis = np.arange(len(indexes_sorted))
        ax[indexing].barh(Y_axis, average_shap_values)
        ax[indexing].set_yticks(Y_axis, indexes_sorted)
        ax[indexing].set_ylabel('Feature Index')
        ax[indexing].set_xlabel('Average SHAP Value ')
        ax[indexing].set_title('Average SHAP Values for ' + season)
        for i, index in enumerate(indexes_sorted):
            if index == 'CASM':
                ax[indexing].barh(Y_axis[i], average_shap_values[i], color='red')
            if index == 'GPP':
                ax[indexing].barh(Y_axis[i], average_shap_values[i], color='green')
            if index == 'RECO':
                ax[indexing].barh(Y_axis[i], average_shap_values[i], color='orange')
    fig.tight_layout(pad=1.0)
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_geographical_SHAP(shap, gt_id, feature, mask):
    """ Plot on 4 maps (seasons) the SHAP values of the added feature to the dataset."""
    # Define the map boundaries
    llcrnrlon = -124
    llcrnrlat = 27
    urcrnrlon = -94
    urcrnrlat = 49
    seasons = ['winter', 'spring', 'summer', 'fall']
    fig, ax = plt.subplots(1, 4, figsize=(25,25))
    ax = ax.ravel()  # Flatten the ax array
    if gt_id == 'contest_tmp2m':
        feature_names = ['rhum_shift44', 'pres_shift44','nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
           'tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86',
           'tmp2m_shift86_anom', 'mei_shift59', 'phase_shift31',
           'sst_2010_1_shift44', 'sst_2010_2_shift44', 'sst_2010_3_shift44',
           'icec_2010_1_shift44', 'icec_2010_2_shift44', 'icec_2010_3_shift44',
           'wind_hgt_10_2010_1_shift44', 'wind_hgt_10_2010_2_shift44', 'CASM', 'GPP', 'RECO', 'elevation', 'ElNino1', 'ElNino34', 'ElNino4' ]
        feature_of_interest  = feature_names.index(feature)
    elif gt_id == 'contest_precip':
        feature_names =  ['rhum_shift44','pres_shift44', 'nmme_wo_ccsm3_nasa','nmme0_wo_ccsm3_nasa',
            'precip_shift43','precip_shift43_anom',
            'tmp2m_shift43','tmp2m_shift43_anom','precip_shift86','precip_shift86_anom','tmp2m_shift86',
            'tmp2m_shift86_anom','mei_shift59','phase_shift31','sst_2010_1_shift44','sst_2010_2_shift44',
            'sst_2010_3_shift44','icec_2010_1_shift44','icec_2010_2_shift44','icec_2010_3_shift44','wind_hgt_10_2010_1_shift44',
            'wind_hgt_10_2010_2_shift44','CASM','GPP','RECO','elevation', 'ElNino1', 'ElNino34', 'ElNino4']
        feature_of_interest  = feature_names.index(feature)
    for indexing, season in enumerate(seasons):
        if indexing == 0:
            shapi = shap[:80,:,:,feature_of_interest]
        if indexing == 1:
            shapi = shap[80:172,:,:,feature_of_interest]
        if indexing == 2:
            shapi = shap[172:265,:,:,feature_of_interest]
        if indexing == 3:
            shapi = shap[265:-9,:,:,feature_of_interest]
        shapi = shapi.mean(axis=0)
        m = Basemap(projection='cyl', llcrnrlon=-125, llcrnrlat=26, urcrnrlon=-93, urcrnrlat=50, resolution='c', ax=ax[indexing])
        m.drawcoastlines()
        m.drawcountries()
        m.drawstates()
        lons, lats = np.meshgrid(np.arange(llcrnrlon, urcrnrlon+1,1), np.arange(llcrnrlat, urcrnrlat+1, 1))
        x, y = m(lons, lats)
        masked_data = np.ma.masked_array(shapi.flatten(), mask=mask.flatten())
        sc = ax[indexing].scatter(x, y, c=masked_data, cmap='jet', s=50)#, vmin=-0.010, vmax=0.003)  # Adjust 'cmap' and 's' as desired
        ax[indexing].set_title('shap value for ' + season)
        cbar = plt.colorbar(sc, ax=ax[indexing], shrink=0.1)
        fig.suptitle('Spatial shap values precipitations anomalies 5/6 for all features for ' + feature, fontsize=30, y=0.58)
    plt.show()
    

#------------------------------------------------------------------------------------------------------------------------------------------------------


def plot_LRP_values(LRP_vector, gt_id, feature):
    """ Plot Layerwise Relevance Propagation values, another explainable AI method, that appears to be less trustworthy than SHAP"""
    if gt_id == 'contest_tmp2m':
        indexes = ['rhum_shift44', 'pres_shift44','nmme_wo_ccsm3_nasa', 'nmme0_wo_ccsm3_nasa',
           'tmp2m_shift43', 'tmp2m_shift43_anom', 'tmp2m_shift86',
           'tmp2m_shift86_anom', 'mei_shift59', 'phase_shift31',
           'sst_2010_1_shift44', 'sst_2010_2_shift44', 'sst_2010_3_shift44',
           'icec_2010_1_shift44', 'icec_2010_2_shift44', 'icec_2010_3_shift44',
           'wind_hgt_10_2010_1_shift44', 'wind_hgt_10_2010_2_shift44', feature]
    elif gt_id == 'contest_precip':
        indexes =  ['rhum_shift44','pres_shift44', 'nmme_wo_ccsm3_nasa','nmme0_wo_ccsm3_nasa',
            'precip_shift43','precip_shift43_anom',
            'tmp2m_shift43','tmp2m_shift43_anom','precip_shift86','precip_shift86_anom','tmp2m_shift86',
            'tmp2m_shift86_anom','mei_shift59','phase_shift31','sst_2010_1_shift44','sst_2010_2_shift44',
            'sst_2010_3_shift44','icec_2010_1_shift44','icec_2010_2_shift44','icec_2010_3_shift44','wind_hgt_10_2010_1_shift44',
            'wind_hgt_10_2010_2_shift44', feature]
    fig, (ax0,ax1) = plt.subplots(nrows=1, ncols=2,figsize=(20, 10))
    idx = np.argsort(LRP_vector)
    idx_abs = np.argsort(abs(LRP_vector))
    indexes_sorted = np.array(indexes)[idx]
    indexes_sorted_abs = np.array(indexes)[idx_abs]
    Y_axis = np.arange(len(indexes_sorted))
    Y_axis_abs = np.arange(len(indexes_sorted_abs))
    LRP_tensor_sorted = LRP_vector[idx]
    LRP_tensor_sorted_abs = LRP_vector[idx_abs]
    ax0.barh(Y_axis, LRP_tensor_sorted,  label = 'LRP' )
    ax1.barh(Y_axis, abs(LRP_tensor_sorted_abs),  label = 'LRP')
    for i, index in enumerate(indexes_sorted):
        if index == feature:
            ax0.barh(Y_axis[i], LRP_tensor_sorted[i], color='red')
    for i, index in enumerate(indexes_sorted_abs):
        if index == feature:
            ax1.barh(Y_axis[i], -LRP_tensor_sorted_abs[i], color='red')

    ax0.set_yticks(Y_axis, indexes_sorted)
    ax0.set_yticklabels(indexes_sorted, rotation='horizontal')
    ax0.set_xlabel("Feature Name")
    ax0.set_ylabel("Average role in prediction")
    ax0.set_title("Average role in prediction with new feature")
    ax0.legend()
    
    ax1.set_yticks(Y_axis_abs, indexes_sorted_abs)
    ax1.set_yticklabels(indexes_sorted_abs, rotation='horizontal')
    ax1.set_xlabel("Feature Name")
    ax1.set_ylabel("Average role in prediction (LRP)")
    ax1.set_title("Average role in prediction with new feature : absolute values (LRP)")
    ax1.legend()

    fig.tight_layout(pad=1.0)
    

#------------------------------------------------------------------------------------------------------------------------------------------------------


def histogram_of_contest_skills(skills_tmp2m_initial, skills_precip_initial, target_horizon): 
    """ plots the repartition of skills for initial data in order to evaluate the model's performances """
    tmp2m_flatten = skills_tmp2m_initial.flatten()
    precip_flatten = skills_precip_initial.flatten()
    bin_edges = np.arange(-1, 1.1, 0.2)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    hist1, _ = np.histogram(tmp2m_flatten, bins=bin_edges)
    hist2, _ = np.histogram(precip_flatten, bins=bin_edges)
    hist1_percentage = (hist1 / len(tmp2m_flatten)) * 100
    hist2_percentage = (hist2 / len(precip_flatten)) * 100

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize = (7,8), sharex=True, sharey=True)
    ax1.set_facecolor('gainsboro')
    ax2.set_facecolor('gainsboro')
    mean1 = tmp2m_flatten.mean()
    mean2 = np.mean(np.array(precip_flatten))
    ax1.axvline(mean1, color='red', linestyle='--', linewidth=2, label='Mean')
    ax2.axvline(mean2, color='red', linestyle='--', linewidth=2, label='Mean')
    ax1.bar(bin_edges[:-1], hist1_percentage, width=0.2, align='edge', color='blue', alpha=0.7, zorder=1)
    ax2.bar(bin_edges[:-1], hist2_percentage, width=0.2, align='edge', color='green', alpha=0.7)
    ax1.grid(color='white', linestyle='--', linewidth=0.5, zorder=0)
    ax2.grid(color='white', linestyle='--', linewidth=0.5)
    if target_horizon == "56w" :
        ax1.text(0.5, 1.1, 'Temperature Week 5-6', transform=ax1.transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='center',
         bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round'))
        ax2.text(0.5, 1.1, 'Precipitation Week 5-6', transform=ax2.transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='center',
         bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round'))
    if target_horizon == "34w" :
        ax1.text(0.5, 1.1, 'Temperature Week 3-4', transform=ax1.transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='center',
         bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round'))
        ax2.text(0.5, 1.1, 'Precipitation Week 3-4', transform=ax2.transAxes,
         fontsize=12, verticalalignment='top', horizontalalignment='center',
         bbox=dict(facecolor='lightgray', edgecolor='gray', boxstyle='round'))
        
    ax1.text(mean1, max(hist1_percentage), f'Mean: {mean1:.2f}', color='red',
            verticalalignment='bottom', horizontalalignment='right', fontsize=12)
    ax2.text(mean2, max(hist2_percentage), f'Mean: {mean2:.2f}', color='red',
         verticalalignment='bottom', horizontalalignment='right', fontsize=12)
    fig.text(0.5, -0.02, 'Skills', ha='center', fontsize = 15)
    fig.text(-0.02, 0.5, 'Percentage', va='center', rotation='vertical', fontsize=15)
    fig.suptitle('Distribution of 2011-2017 skills for Neural Network model', fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
