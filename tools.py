import torch.nn.functional as F
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torch as th
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
from pytorch_forecasting.metrics.base_metrics import MultiHorizonMetric
device = th.device("cuda" if th.cuda.is_available() else "cpu")
import numpy as np
from torch.utils.data import TensorDataset
from math import *
import pandas as pd
pd.set_option('display.precision', 12)
from experiments_util import *
# Load functionality for fitting and predicting
#from fit_and_predict import *
# Load functionality for evaluation
from skill import *
# Load functionality for stepwise regression
from stepwise_util import *
from tools import *
import warnings
import torch.optim as optim
import shap



#------------------------------------------------------------------------------------------------------------------------------------------------------

def restriction_mask(mask):
    """Aims at deleting the pixels in the coastal areas or at the border to avoid variations due to the oceanic weather variations"""
    
    dilated_mask = binary_dilation(mask, iterations=2)

    # Find the difference between the dilated mask and the original mask, it gives the pixels that were changed from False to True near the border
    border_diff = dilated_mask & ~np.array(mask)

    # Customize some pixels
    border_diff[15,0] = True
    border_diff[16,0] = True
    border_diff[14,1] = True
    border_diff[15,1] = True
    border_diff[16,1] = True
    border_diff[17,1] = True

    # Combine the border_diff with the original mask to get the final mask with fewer True values
    mask_no_border = mask | border_diff
    return(mask_no_border)

#------------------------------------------------------------------------------------------------------------------------------------------------------

def normalization(data, reversed_mask, is_mask, target_prediction, initial_dataset, index_feature, type_norm, indexes, gt_id):
    """Aims at normalizing the data around 0. Another normalization method can be used when uncommenting some data (normalization between 0 and 1).
    
    The indexes_used can be modified depending of the 
    
    
    """
    #print('There is a Mask applied :', is_mask)
    if is_mask == False: #using a rectangle without NaN values
        #### Biggest practical (for Maxpoolig and upsample) rectangle inside the Western USA (without touching borders)
        data = data[:,8:20,4:24,:]
        tensor_used = data.clone()
        
        #### Normalization #1
        min_value = tensor_used.min(dim=0)[0].min(dim=0)[0].min(dim=0)[0]
        max_value = tensor_used.max(dim=0)[0].max(dim=0)[0].max(dim=0)[0]
        if type_norm == "min_max":
            normalized_tensor = (tensor_used - min_value) / (max_value - min_value)

        #### Normalization #2
        mean = tensor_used.mean(dim=(0,1,2))
        std = tensor_used.std(dim=(0,1,2))
        if type_norm == "mean_std":
            normalized_tensor = (tensor_used - mean) / (std + 1e-7)
        
        target_data = normalized_tensor[:,:,:,target_prediction].to(device)
        
    elif is_mask == True: #using all the US map with points with NaN values
        tensor_used = data.clone()
        # Reshape the mask to match the shape of the data tensor
        reshaped_mask = reversed_mask.unsqueeze(0).unsqueeze(3)
        # Apply the mask to the data tensor
        reshaped_mask = reshaped_mask.cpu()
        masked_data = tensor_used * reshaped_mask
        clone = masked_data.clone()
        
        #normalization #1
        min_value = clone.min(dim=0)[0].min(dim=0)[0].min(dim=0)[0]
        max_value = clone.max(dim=0)[0].max(dim=0)[0].max(dim=0)[0]
        if type_norm == "min_max":
            normalized_tensor = (masked_data - min_value) / (max_value - min_value)
            normalized_tensor = normalized_tensor * reshaped_mask
        
        #normalization #2
        mean = clone.mean(dim=(0,1,2))
        std = clone.std(dim=(0,1,2))
        if type_norm == "mean_std":
            normalized_tensor = (clone - mean) / (std + 1e-7)
        
        target_data = normalized_tensor[:,:,:,target_prediction].to(device) * reversed_mask.unsqueeze(0).to(device) #to be sure it has the same values to 0
        
    #store values to denormalize
    min_value_temp_anom = min_value[target_prediction]
    max_value_temp_anom = max_value[target_prediction]
    mean_value_temp_anom = mean[target_prediction]
    std_value_temp_anom = std[target_prediction]
    
    #Iitial data from RODEO
    input_data = th.concat((normalized_tensor[:,:,:,3:5],normalized_tensor[:,:,:,8:10]),axis=3)
    indexes_used = np.concatenate((indexes[3:5],indexes[8:10]))
    
    if gt_id == "contest_tmp2m" :
        input_data = th.concat((input_data,normalized_tensor[:,:,:,12:26]),axis=3)
        indexes_used = np.concatenate((indexes_used,indexes[12:26]))
    if gt_id == "contest_precip" :
        input_data = th.concat((input_data,normalized_tensor[:,:,:,13:31]),axis=3)
        indexes_used = np.concatenate((indexes_used,indexes[13:31]))
    
    #####new data you want to consider in the dataset (uncomment and modify the number as you wish)
    if initial_dataset == False :
        input_data = th.concat((input_data,normalized_tensor[:,:,:,index_feature:index_feature+1]),axis=3) #new feature you want to add
        indexes_used = np.concatenate((indexes_used,indexes[index_feature:index_feature+1])) #new feature you want to add
    
    if type_norm == "min_max":
        return(normalized_tensor, input_data, target_data, indexes_used, min_value_temp_anom, max_value_temp_anom)
    if type_norm == "mean_std":
        return(normalized_tensor, input_data, target_data, indexes_used, mean_value_temp_anom, std_value_temp_anom)

#------------------------------------------------------------------------------------------------------------------------------------------------------

class linear_model(nn.Module):
    """Very basic model composed of a linear layer along the features dimension.
    Other examples of models (not stable or efficient cans also be found in this file.
    """
    def __init__(self, model_input, mask):
        super(linear_model, self).__init__()
        self.fc1 = nn.Linear(model_input, 1)
        self.reverse_mask = ~mask.to(device)
    def forward(self, x):
        batch_size_value = x.size(0)
        lat_lon_shape = x.size()[1:3] 
        nb_feat = x.size(3)
        x = self.fc1(x)
        x=x.reshape(batch_size_value, lat_lon_shape[0], lat_lon_shape[1])
        return(x)
    
    
    def explain(self, x): #Layerwise Relevance Propagation Analysis
        batch_size = x.size(0)
        x = x.permute(0, 3, 1, 2)
        x = x*self.reverse_mask
        x = x.permute(0,2,3,1)
        x.requires_grad_()
        output = self.forward(x) # Forward pass
        relevance_scores = th.zeros_like(output*self.reverse_mask) # Set the relevance score of the output prediction to 1
        relevance_scores.fill_(1)        
        output.backward(gradient=relevance_scores) # Backward pass with relevance propagation
        relevance_scores = x.grad # Compute the relevance scores for the input features
        return(relevance_scores)
    
#------------------------------------------------------------------------------------------------------------------------------------------------------
    
class QuantileLoss(MultiHorizonMetric):
    """
    Quantile loss, i.e. a quantile of ``q=0.5`` will give half of the mean absolute error as it is calculated as
    Defined as ``max(q * (y-y_pred), (1-q) * (y_pred-y))``
    The use of this quantile loss might, once optimized, increase the amplitude of predictions. Otherwise a simple MSE loss is used.
    """
    def __init__(self, quantiles: List[float] = [0.02, 0.1, 0.25, 0.5, 0.75, 0.9, 0.98], **kwargs):
        """ Quantile loss
        Args:
            quantiles: quantiles for metric """
        super().__init__(quantiles=quantiles, **kwargs)

    def loss(self, y_pred: Dict[str, th.Tensor], target: th.Tensor) -> th.Tensor:
        """  Calculate the quantile loss for each quantile level and return the sum.
        Args:
            y_pred: Dictionary of predicted values for each quantile level
            target: Ground truth target values
        Returns:
            torch.Tensor: Quantile loss as a single number for backpropagation """
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - y_pred[i]  # Access predicted values using quantile index
            loss = th.max((q - 1) * errors, q * errors)
            losses.append(loss)   
        return th.mean(th.stack(losses))
    
#------------------------------------------------------------------------------------------------------------------------------------------------------

def trainer(model, train_loader, val_loader, criterion, optimizer, num_epochs, reversed_mask, is_mask, regularization):
    """ training function """
    train_losses=[]
    valid_losses=[]
    l1_lambda = 0.01
    l2_lambda = 0.01
    for epoch in range(num_epochs):
        trainloss=0
        model.train()
        #tic()
        for inputs, targets in train_loader :
            inputs = inputs.to(device)
            targets = targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            
            if is_mask == False: 
                if regularization == "None":
                    loss = criterion(outputs, targets)
                
                if regularization == "L1":
                    regularization_term = 0
                    for param in model.parameters():
                        regularization_term += th.sum(th.abs(param))
                    loss = criterion(outputs,targets) + l1_lambda * regularization_term
                
                if regularization == "L2":
                    regularization_term = 0
                    for param in model.parameters():
                        regularization_term += th.sum(th.pow(param, 2))
                    loss = criterion(outputs, targets) + l2_lambda * regularization_term
            
            elif is_mask == True:
                reversed_mask = reversed_mask.to(device)
                masked_predictions = outputs * reversed_mask
                masked_targets = targets * reversed_mask
                reshaped_predictions = masked_predictions.flatten(start_dim=1).to(device)
                reshaped_targets = masked_targets.flatten(start_dim=1).to(device)
                mask_gpu = reversed_mask.flatten().to(device)
                pertinent_predictions = reshaped_predictions[:,mask_gpu]
                pertinent_targets = reshaped_targets[:,mask_gpu]
                
                if regularization == "None":
                    loss = criterion(outputs, targets)
                
                if regularization == "L1":
                    regularization_term = 0
                    for param in model.parameters():
                        regularization_term += th.sum(th.abs(param))
                    loss = criterion(outputs,targets) + l1_lambda * regularization_term
                
                if regularization == "L2":
                    regularization_term = 0
                    for param in model.parameters():
                        regularization_term += th.sum(th.pow(param, 2))
                    loss = criterion(outputs, targets) + l2_lambda * regularization_term
            trainloss+=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        train_losses.append(100*trainloss/len(train_loader))

        model.eval()  
        with th.no_grad():
            total_loss = 0.0
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.to(device)
                outputs =  model(inputs)
                if is_mask == False: 
                    if regularization == "None":
                        loss = criterion(outputs, targets)
                
                    if regularization == "L1":
                        regularization_term = 0
                        for param in model.parameters():
                            regularization_term += th.sum(th.abs(param))
                        loss = criterion(outputs,targets) + l1_lambda * regularization_term
                
                    if regularization == "L2":
                        regularization_term = 0
                        for param in model.parameters():
                            regularization_term += th.sum(th.pow(param, 2))
                        loss = criterion(outputs, targets) + l2_lambda * regularization_term
                    
                elif is_mask == True:
                    reversed_mask = reversed_mask.to(device)
                    masked_predictions = outputs * reversed_mask
                    masked_targets = targets * reversed_mask
                    reshaped_predictions = masked_predictions.flatten(start_dim=1).to(device)
                    reshaped_targets = masked_targets.flatten(start_dim=1).to(device)
                    mask_gpu = reversed_mask.flatten().to(device)
                    pertinent_predictions = reshaped_predictions[:,mask_gpu]
                    pertinent_targets = reshaped_targets[:,mask_gpu]
                    
                    if regularization == "None":
                        loss = criterion(outputs, targets)
                
                    if regularization == "L1":
                        regularization_term = 0
                        for param in model.parameters():
                            regularization_term += th.sum(th.abs(param))
                        loss = criterion(outputs,targets) + l1_lambda * regularization_term
                
                    if regularization == "L2":
                        regularization_term = 0
                        for param in model.parameters():
                            regularization_term += th.sum(th.pow(param, 2))
                        loss = criterion(outputs, targets) + l2_lambda * regularization_term

                total_loss += loss.item() 
            avg_valid_loss = 100*total_loss / len(val_loader)
        valid_losses.append(avg_valid_loss)    
        print("Number of Epoch :",format(epoch+1), "\t train loss :", format(train_losses[-1]), "\t valid loss :", format(valid_losses[-1]))
        #toc()
        
    #plt.figure(figsize = (9, 1.5))
    #ax= plt.subplot(1, 2, 1)
    #plt.plot(train_losses)
    #plt.title('train losses')
    #ax= plt.subplot(1, 2, 2)
    #plt.plot(valid_losses)
    #plt.title('valid losses')
    #plt.show()
    return(train_losses[-1],valid_losses[-1])

#------------------------------------------------------------------------------------------------------------------------------------------------------

def evaluate(test_data, test_targets, target_year, time_test, is_mask, reversed_mask, type_of_norm, model, datetime_objects, DataLoader, batch_size, value1, value2): 
    """ Returns predictions and targets tensors over a chosen target_year"""
    model.eval()
    predictions = []
    targets_list = []
    predictions_loss = []
    targets_list_loss = []
    dates_list = []

    start_date = [dt for dt in datetime_objects if dt.year == target_year]
    indexes_year = [i for i, dt in enumerate(time_test) if dt in start_date] #extract the indexes of the year of interest
    test_dataset = TensorDataset(test_data[indexes_year], test_targets[indexes_year])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
    with th.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predicted_temperatures = model(inputs)
            predictions.append(predicted_temperatures.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())
            if is_mask == True : 
                masked_predictions = predicted_temperatures.to(device) * reversed_mask.to(device)
                masked_targets = targets.to(device) * reversed_mask.to(device)
                reshaped_predictions = masked_predictions.flatten(start_dim=1).to(device)
                reshaped_targets = masked_targets.flatten(start_dim=1).to(device)
                mask_gpu = reversed_mask.flatten().to(device)
                pertinent_predictions = reshaped_predictions[:,mask_gpu]
                pertinent_targets = reshaped_targets[:,mask_gpu]
                predictions_loss.append(pertinent_predictions.detach().cpu().numpy())
                targets_list_loss.append(pertinent_targets.detach().cpu().numpy())    
    predictions = np.concatenate(predictions)
    targets_array = np.concatenate(targets_list)
    if type_of_norm == "min_max": #opposite of normalization NORM1
        predictions = th.tensor(predictions) * (value2 - value1) + value1
        targets_array = th.tensor(targets_array) * (value2 - value1) + value1
    if type_of_norm == "mean_std": #NORM2
        predictions = th.tensor(predictions) * (value2+1e-7) + value1
        targets_array = th.tensor(targets_array) * (value2+1e-7) + value1
        
    if is_mask == False:         
        predictions2 = predictions.view(predictions.shape[0], -1)
        targets_array2 = targets_array.view(targets_array.shape[0], -1)
        mse = mean_squared_error(targets_array2, predictions2)
        rmse = sqrt(mse)
        return(predictions, targets_array, rmse, start_date)
    elif is_mask == True:
        predictions1_loss = np.concatenate(predictions_loss)
        targets_list1_loss = np.concatenate(targets_list_loss)
        if type_of_norm == "min_max":
            predicted_tensor = th.from_numpy(predictions1_loss)*(value2 - value1) + value1
            predicted_test = predicted_tensor.view(predicted_tensor.shape[0], -1)
            target_tensor = th.from_numpy(targets_list1_loss)*(value2 - value1) + value1
        if type_of_norm == "mean_std":
            predicted_tensor = th.from_numpy(predictions1_loss) * (value2+1e-7) + value1
            predicted_test = predicted_tensor.view(predicted_tensor.shape[0], -1)
            target_tensor = th.from_numpy(targets_list1_loss) * (value2+1e-7) + value1
        target_test = target_tensor.view(target_tensor.shape[0], -1)
        mse = mean_squared_error(target_test, predicted_test)
        rmse = sqrt(mse)
        return(predictions, targets_array, rmse, start_date)
        
#------------------------------------------------------------------------------------------------------------------------------------------------------

#Provide the skills for each year between 2011 and 2017

def get_skill_fn(test_data, test_targets, time_test, is_mask, reversed_mask, type_of_norm, model, datetime_objects, DataLoader, batch_size, value1, value2):
    """ Calculates the skills for all the years betwen 2011 ad 2017 to evaluate the model's performances"""
    reversed_mask = reversed_mask.to(device)
    years=[2011+i for i in range(7)]
    result3=pd.DataFrame()
    result3['year']=years
    skills=[]
    for year in years:
        predictions, targets_array, mse, start_date = evaluate(test_data, test_targets, year, time_test, is_mask, reversed_mask, type_of_norm, model, datetime_objects, DataLoader, batch_size, value1, value2)
        if is_mask == False:
            reshaped_predictions = predictions.reshape(predictions.shape[0],predictions.shape[1]*predictions.shape[2]).to(device)
            reshaped_targets = targets_array.reshape(targets_array.shape[0],targets_array.shape[1]*targets_array.shape[2]).to(device)
            pertinent_predictions = reshaped_predictions.permute(1,0)
            pertinent_targets = reshaped_targets.permute(1,0)
        elif is_mask == True:   
            masked_predictions = predictions.to(device) * reversed_mask
            masked_targets = targets_array.to(device) * reversed_mask
            reshaped_predictions = masked_predictions.flatten(start_dim=1).to(device)
            reshaped_targets = masked_targets.flatten(start_dim=1).to(device)
            mask_gpu = reversed_mask.flatten().to(device)
            pertinent_predictions = reshaped_predictions[:,mask_gpu]
            pertinent_targets = reshaped_targets[:,mask_gpu]
            pertinent_predictions = pertinent_predictions.permute(1,0)
            pertinent_targets = pertinent_targets.permute(1,0)
    
        flattened_predictions = pertinent_predictions.flatten()
        flattened_targets = pertinent_targets.flatten()
    
        numpy_array_flat_pred = flattened_predictions.cpu().numpy()
        numpy_array_flat_targ = flattened_targets.cpu().numpy()
    
        df_flat_pred = pd.DataFrame(numpy_array_flat_pred)
        df_flat_targ = pd.DataFrame(numpy_array_flat_targ)
    
        concatenated_dates = start_date * pertinent_targets.shape[0] #514 = number of interesting points

        start_date = pd.DataFrame(concatenated_dates)
        results = pd.concat((start_date, df_flat_targ, df_flat_pred),axis=1)
        results.columns = ['start_date', 'targets_array', 'predictions']
        skill=get_col_skill(results,'targets_array','predictions',date_col='start_date')
        skills.append(skill)
    result3['skill']=skills
    return(result3, mse)


#------------------------------------------------------------------------------------------------------------------------------------------------------


from mpl_toolkits import mplot3d
def LRP_method2(test_data, test_targets, time_test, indexes_used, datetime_objects, DataLoader, batch_size, model, mask):
    """ Calculates Layerwise Relevance Propagation values (explainable AI method) """
    years=[2011+i for i in range(7)]
    LRP_tensor = th.zeros(len(years),test_data.shape[1],test_data.shape[2],test_data.shape[3])
    occurence_max_list = []
    occurence_mean_list = []
    j=0
    for target_year in years:
        start_date = [dt for dt in datetime_objects if dt.year == target_year]
        indexes_year = [i for i, dt in enumerate(time_test) if dt in start_date] 
        test_dataset = TensorDataset(test_data[indexes_year], test_targets[indexes_year])
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
        with th.no_grad():
            test_LRP_data = th.tensor([]).to(device)
            for inputs, targets in test_loader : 
                inputs = inputs.to(device)
                test_LRP_data = th.concat((test_LRP_data,inputs),dim=0)
        model.train()
        
        # Obtain the relevance scores using LRP
        relevant_scores = model.explain(test_LRP_data)
        relevant_scores = np.array(relevant_scores.cpu())
        reshaped_mask = mask.repeat(relevant_scores.shape[0],relevant_scores.shape[3],1,1)
        reshaped_mask = reshaped_mask.permute(0,2,3,1)
        masked_data_relevant_scores = np.ma.masked_array(relevant_scores, mask=reshaped_mask)
        LRP_tensor[j,:,:,:]=th.tensor(masked_data_relevant_scores.mean(axis=0))

        # Calculate the number of times each feature appears as the best feature
        best_feature_counts = np.argmax(masked_data_relevant_scores, axis=-1)

        occurrence_2D = np.zeros(indexes_used.shape)
        for i in range(best_feature_counts.shape[0]):
            # Count the occurrence of each element in the array
            unique_elements, counts =np.unique(best_feature_counts[i], return_counts=True)

            # Print the unique elements and their corresponding counts
            for element, count in zip(unique_elements, counts):
                occurrence_2D[element] += count
            
        ###Mean occurences at each date
        output_array = np.mean(masked_data_relevant_scores, axis=(1, 2)) #mean over sspatial coordinates
        output_array = output_array
        mean_output_array = np.mean(output_array, axis=0)
        
        occurence_max_list.append(occurrence_2D)
        occurence_mean_list.append(mean_output_array)
        j+=1
        
    return(np.array(occurence_max_list),np.array(occurence_mean_list), LRP_tensor)


#------------------------------------------------------------------------------------------------------------------------------------------------------


def one_training(regularization, normalized_tensor, input_data, target_data, datetime_objects, TensorDataset, DataLoader, batch_size):

    """This is where data is split up between train, validation and test set. Preprocess the data before training. """

    Ntime, Nlat, Nlon, Nfeature = normalized_tensor.shape[0], normalized_tensor.shape[1], normalized_tensor.shape[2], normalized_tensor.shape[3]
    input_shape = (Ntime, Nlat, Nlon, Nfeature)
    print(input_shape)
    model_input = input_data.shape[3]

    # Determine the sizes for train, validation, and test sets based on the time dimension
    time_steps = input_data.shape[0]
    train_size = int(0.75 * time_steps)
    val_size = int(0.1 * (time_steps - train_size))
    test_size = time_steps - train_size - val_size
    #print(test_size)

    # Split the dataset based on the time dimension
    train_data = input_data[:train_size, :, :, :]
    train_targets = target_data[:train_size, :, :]
    val_data = input_data[train_size:train_size+val_size, :, :, :]
    val_targets = target_data[train_size:train_size+val_size, :, :]
    test_data = input_data[train_size+val_size:train_size+val_size+test_size, :, :, :]
    test_targets = target_data[train_size+val_size:train_size+val_size+test_size, :, :]
    #print('test_data',test_data.shape)
    #print('test_targets',test_targets.shape)
    time_test = datetime_objects[-test_size:]

    # Convert the split data into TensorDatasets
    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    test_dataset = TensorDataset(test_data, test_targets)

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    return(train_loader, val_loader, test_loader, time_test, model_input, test_data, test_targets)


#------------------------------------------------------------------------------------------------------------------------------------------------------


def visualization_one_training(predictions, targets_array, latitude, longitude, gt_id):
    """ plots for one chosen point the predictions and targets during one entire year """
    plt.plot(targets_array[:,latitude,longitude], label='Actual Targets')
    plt.plot(predictions[:,latitude,longitude], label='Model Predictions')
    plt.xlabel('Time')
    if gt_id == "contest_tmp2m" : 
        plt.ylabel('Temperature anomalies')
    if gt_id == "contest_precip" : 
        plt.ylabel('Precipitation anomalies')
    plt.title('Actual Targets vs Model Predictions')
    plt.legend()
    plt.show()

    fig, ax1 = plt.subplots()
    # Plot the actual targets on the first axes
    ax1.plot(targets_array[:, latitude, longitude], label='Actual Targets', color='b')
    ax1.set_xlabel('Time')
    if gt_id == "contest_tmp2m" : 
        ax1.set_ylabel('Temperature anomalies (Celsius)', color='b')
    if gt_id == "contest_precip" : 
        ax1.set_ylabel('Precipitation anomalies', color='b')
    ax1.tick_params('y', colors='b')

    # Create the second set of axes
    ax2 = ax1.twinx()

    # Plot the model predictions on the second axes
    ax2.plot(predictions[:, latitude, longitude], label='Model Predictions', color='r')
    if gt_id == "contest_tmp2m" : 
        ax2.set_ylabel('Temperature anomalies (Normalized)', color='r')
    if gt_id == "contest_precip" : 
        ax2.set_ylabel('Precipitation anomalies (Normalized)', color='r')
        
    ax2.tick_params('y', colors='r')
    plt.title('Actual Targets vs Model Predictions')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()
    

#------------------------------------------------------------------------------------------------------------------------------------------------------
    

def visualization_skills(test_data,test_targets,time_test,is_mask, reversed_mask, type_of_norm,model, datetime_objects, DataLoader, batch_size, value1, value2):
    """ Plots the  skills over the years 2011-2017"""
    result3, mse = get_skill_fn(test_data,test_targets,time_test,is_mask, reversed_mask, type_of_norm,model, datetime_objects, DataLoader, batch_size, value1, value2)
    results = np.array(result3['skill'])
    print(result3)
    print(results.mean())
    plt.plot(result3['year'],result3['skill'],label='NN')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Skill')
    plt.show()

    
#------------------------------------------------------------------------------------------------------------------------------------------------------


def compute_moving_average(tensor, window_size=30):
    """ preprocess for visualization moving average. Averages over 'window_size' days """
    output_tensor = th.zeros((tensor.shape[0], tensor.shape[1], tensor.shape[2]))
    
    # For the first 29 days, copy the values as they are
    output_tensor[:window_size - 1] = tensor[:window_size - 1]
    #print(output_tensor[:,7,7])

    # Calculate the moving average for the rest of the days
    for i in range(window_size - 1, tensor.shape[0]):
        # Compute the average of the current day and the previous 29 days along the first dimension
        average = th.mean(tensor[i - window_size + 1:i + 1], dim=0)
        output_tensor[i] = average

    return output_tensor


#------------------------------------------------------------------------------------------------------------------------------------------------------


def visualization_moving_average(predictions, targets_array, latitude, longitude, gt_id, window_size):
    """ similar to visualization on training but this date, each point has been averaged over x days to get rid off weather instability """
    predictions_flat = compute_moving_average(predictions, window_size)
    targets_flat = compute_moving_average(targets_array, window_size)

    plt.plot(targets_flat[:,latitude,longitude], label='Actual Targets')
    plt.plot(predictions_flat[:,latitude,longitude], label='Model Predictions')
    plt.xlabel('Time')
    if gt_id == "contest_tmp2m":
        plt.ylabel('Temperature Anomalies')
    if gt_id == "contest_precip":
        plt.ylabel('Precipitation anomalies')
    plt.title('Actual Targets vs Model Predictions averaged over ' + str(window_size) + ' days')
    plt.legend()
    plt.show()

    fig, ax1 = plt.subplots()
    ax1.plot(targets_flat[:, latitude, longitude], label='Actual Targets', color='b')
    ax1.set_xlabel('Time')
    
    ax1.tick_params('y', colors='b')
    ax2 = ax1.twinx()
    ax2.plot(predictions_flat[:, latitude, longitude], label='Model Predictions', color='r')
    if gt_id == "contest_tmp2m":
        ax1.set_ylabel('Temperature anomalies ', color='b')
        ax2.set_ylabel('Temperature anomalies (Zoom)', color='r')
    if gt_id == "contest_precip":
        ax1.set_ylabel('Precip anomalies ', color='b')
        ax2.set_ylabel('Precip anomalies (Zoom)', color='r')
    ax2.tick_params('y', colors='r')
    plt.title('Actual Targets vs Model Predictions averaged over ' + str(window_size) + ' days')
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.show()


#------------------------------------------------------------------------------------------------------------------------------------------------------


def multiple_trainings(regularization, number_of_trainings, type_of_norm, data, reversed_mask, is_mask, target_prediction, initial_dataset, index_feature, indexes, datetime_objects, DataLoader, batch_size, learning_rate, num_epochs, value1, value2, mask, indexes_used, target_year, cache_dir, name_file_averaged_training, gt_id):
    """ Averages the results oveer x trainings to gain in stability """
    trainloss_store = []
    mean_skill_store = []
    all_skill_store = []
    occurence_mean_store = []
    stock_all_pred = []
    stock_all_target = []

    warnings.filterwarnings('ignore',category=RuntimeWarning)

    for iteration in range(number_of_trainings):
        if type_of_norm == "min_max":
            normalized_tensor, input_data, target_data, indexes_used, min_value, max_value = normalization(data, reversed_mask, is_mask, target_prediction, initial_dataset, index_feature, type_of_norm, indexes, gt_id)
        if type_of_norm == "mean_std":
            normalized_tensor, input_data, target_data, indexes_used, mean_value, std_value = normalization(data, reversed_mask, is_mask, target_prediction, initial_dataset, index_feature, type_of_norm, indexes, gt_id)
        
        time_steps = input_data.shape[0]
        train_size = int(0.75 * time_steps)
        val_size = int(0.1 * (time_steps - train_size))
        test_size = time_steps - train_size - val_size
        model_input = input_data.shape[3]
        
        # Split the dataset based on the time dimension
        train_data = input_data[:train_size, :, :, :]
        train_targets = target_data[:train_size, :, :]
        val_data = input_data[train_size:train_size+val_size, :, :, :]
        val_targets = target_data[train_size:train_size+val_size, :, :]
        test_data = input_data[train_size+val_size:train_size+val_size+test_size, :, :, :]
        test_targets = target_data[train_size+val_size:train_size+val_size+test_size, :, :]
        time_test = datetime_objects[-test_size:]

        train_dataset = TensorDataset(train_data, train_targets)
        val_dataset = TensorDataset(val_data, val_targets)
        test_dataset = TensorDataset(test_data, test_targets)

        # Create data loaders for train, validation, and test sets
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    
        model = linear_model(model_input, mask).to(device)
        criterion = nn.MSELoss() #MSE or  QuantileLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        tic()
        train_loss, valid_loss = trainer(model, train_loader, val_loader, criterion, optimizer, num_epochs, reversed_mask, is_mask, regularization)
        print('end of training',iteration)
        print("\t train loss :", format(train_loss), "\t valid loss :", format(valid_loss))
        trainloss_store.append(train_loss)
        result3, mse = get_skill_fn(test_data, test_targets, time_test, is_mask, reversed_mask, type_of_norm, model, datetime_objects, DataLoader, batch_size, value1, value2)
        print('rmse',mse)
        results = np.array(result3['skill'])
        all_skill_store.append(np.array(result3))
        mean_skill_store.append(results.mean())
        print(results)
        plt.plot(result3['year'],result3['skill'],label='NN')
        plt.legend()
        plt.xlabel('Year')
        plt.xticks(np.arange(result3['year'].min(), result3['year'].max()+1, 1.0))
        plt.ylabel('Skill')

        plt.show()
        occurrence_max, occurence_mean, LRP_tensor= LRP_method2(test_data, test_targets, time_test, indexes_used, datetime_objects, DataLoader,  batch_size, model, mask)
        occurence_mean_store.append(occurence_mean)
    
        #####get skill 3D
        stock_pred = []
        stock_target = []
    
        reversed_mask = reversed_mask.to(device)
        #years=[2011+i for i in range(8)]
        years=[2011+i for i in range(7)]
        lat =[27.0, 28.0, 29.0, 30.0, 31.0, 32.0, 33.0, 34.0, 35.0, 36.0, 37.0, 38.0, 39.0, 40.0, 41.0, 42.0, 43.0, 44.0, 45.0, 46.0, 47.0, 48.0, 49.0]
        lon = [236.0, 237.0, 238.0, 239.0, 240.0, 241.0, 242.0, 243.0, 244.0, 245.0, 246.0, 247.0,
           248.0, 249.0, 250.0, 251.0, 252.0, 253.0, 254.0, 255.0, 256.0, 257.0, 258.0, 259.0, 260.0, 261.0, 262.0, 263.0, 264.0, 265.0, 266.0]
    
        result3=pd.DataFrame()
        result3['year'] = years
        seasons = ['winter','spring','summer','fall']
        skills=th.zeros((len(years),len(seasons),reversed_mask.shape[0],reversed_mask.shape[1]))
        for year in years:
            stock_season_pred=[]
            stock_season_target=[]
            #for season, indexing in enumerate(seasons):
                #print(year, indexing)
            t = time.time()
            predictions, targets_array, mse, start_date = evaluate(test_data, test_targets, target_year, time_test, is_mask, reversed_mask, type_of_norm, model, datetime_objects, DataLoader, batch_size, value1, value2)
            if is_mask == False:
                reshaped_predictions = predictions.reshape(predictions.shape[0],predictions.shape[1]*predictions.shape[2]).to(device)
                reshaped_targets = targets_array.reshape(targets_array.shape[0],targets_array.shape[1]*targets_array.shape[2]).to(device)
                pertinent_predictions = reshaped_predictions.permute(1,0)
                pertinent_targets = reshaped_targets.permute(1,0)
            elif is_mask == True:   
                masked_predictions = predictions.to(device) * reversed_mask
                masked_targets = targets_array.to(device) * reversed_mask
                #print('a',masked_predictions.shape)
                reshaped_predictions = masked_predictions.flatten(start_dim=1).to(device)
                reshaped_targets = masked_targets.flatten(start_dim=1).to(device)
                #print('b',reshaped_predictions.shape)
                mask_gpu = reversed_mask.flatten().to(device)
                #print('c',mask_gpu.shape)
                #pertinent_predictions = reshaped_predictions[:,mask_gpu]
                #pertinent_targets = reshaped_targets[:,mask_gpu]
                
                #print('1',reshaped_predictions.shape)
                pertinent_predictions = reshaped_predictions.permute(1,0)
                pertinent_targets = reshaped_targets.permute(1,0)
                #print('2',pertinent_predictions.shape)
            
            pertinent_predictions = pertinent_predictions.reshape(reversed_mask.shape[0],reversed_mask.shape[1],pertinent_predictions.shape[1]) #lat,lon,date
            pertinent_targets = pertinent_targets.reshape(reversed_mask.shape[0],reversed_mask.shape[1],pertinent_targets.shape[1]) #lat,lon,date
    
            numpy_array_flat_pred = pertinent_predictions.cpu()#.numpy()
            numpy_array_flat_targ = pertinent_targets.cpu()#.numpy()
        
            stock_pred.append(numpy_array_flat_pred)
            stock_target.append(numpy_array_flat_targ)
            
            #stock_season_pred.append(stock_pred)
            #stock_season_target.append(stock_target)

        stock_all_pred.append(stock_pred)
        stock_all_target.append(stock_target)

    mean_tensors_pred = []
    for year_idx in range(len(stock_all_pred[0])):
        tensors_year_pred = [training[year_idx] for training in stock_all_pred]
        #mean_tensors_season_pred = []
        #for season_idx in range(len(stock_all_pred[0][0])):
            #tensors_season_pred = [training[season_idx] for training in tensors_year_pred]
            #stacked_tensor_season_pred = th.stack(tensors_season_pred)
            #mean_tensor_season_pred = th.mean(stacked_tensor_season_pred, dim=0)
            #mean_tensors_season_pred.append(mean_tensor_season_pred)
        stacked_tensor_pred = th.stack(tensors_year_pred)
        mean_tensor_pred = th.mean(stacked_tensor_pred, dim=0)
        mean_tensors_pred.append(mean_tensor_pred)
    print('mean',len(mean_tensors_pred))

    mean_tensors_target = []
    for year_idx in range(len(stock_all_target[0])):
        tensors_year_target = [training[year_idx] for training in stock_all_target]
        #mean_tensors_season_target = []
        #for season_idx in range(len(stock_all_target[0][0])):
        #    tensors_season_target = [training[season_idx] for training in tensors_year_target]
        #    stacked_tensor_season_target = th.stack(tensors_season_target)
        #    mean_tensor_season_target = th.mean(stacked_tensor_season_target, dim=0)
        #    mean_tensors_season_target.append(mean_tensor_season_target)
        stacked_tensor_target = th.stack(tensors_year_target)
        mean_tensor_target = th.mean(stacked_tensor_target, dim=0)
        mean_tensors_target.append(mean_tensor_target)
    print('mean',len(mean_tensors_target))
    
    count = 0
    for year in years:
        for season, indexing in enumerate(seasons): 
            print('(year, season) = ', year, indexing)
            if season==0 : #winter
                numpy_array_flat_pred = mean_tensors_pred[count][:,:,:80] #1
                numpy_array_flat_targ = mean_tensors_target[count][:,:,:80] #1
                date = start_date[:80]
            elif season==1 : #spring
                numpy_array_flat_pred = mean_tensors_pred[count][:,:,80:172] #2
                numpy_array_flat_targ = mean_tensors_target[count][:,:,80:172] #2
                date = start_date[80:172]
            elif season==2 : #summer
                numpy_array_flat_pred = mean_tensors_pred[count][:,:,172:265] #3
                numpy_array_flat_targ = mean_tensors_target[count][:,:,172:265] #3
                date = start_date[172:265]
            elif season==3 : #fall
                numpy_array_flat_pred = mean_tensors_pred[count][:, :,265:-9] #4
                numpy_array_flat_targ = mean_tensors_target[count][:,:,265:-9] #4
                date = start_date[265:-9]
            for i in range(reversed_mask.shape[0]):
                for j in range(reversed_mask.shape[1]):
                    df_flat_pred = pd.DataFrame(numpy_array_flat_pred[i,j,:])
                    df_flat_targ = pd.DataFrame(numpy_array_flat_targ[i,j,:])
                    date_df = pd.DataFrame(date)
                    results = pd.concat((date_df, df_flat_targ, df_flat_pred),axis=1)
                    results.columns = ['start_date', 'targets_array', 'predictions']
                    skill=get_col_skill(results,'targets_array','predictions',date_col='start_date')
                    skills[count,season,i,j] = skill
            #masked_data1 = np.ma.masked_array(skills, mask=mask.repeat(7,1,1))
            #mean1 = masked_data1.mean(axis=(1,2))
        print("skills_shape",  skills.shape)
        count += 1

    # if cache_dir doesn't exist, create it
    if not os.path.isdir(cache_dir):
        os.makedirs(cache_dir)

    # Filenames for data file to be stored in cache_dir
    data_file = os.path.join(
        cache_dir, name_file_averaged_training)

    print("Saving multiarrays features to " + data_file)
    th.save(skills, data_file)

    print("Finished generating data matrix.")


    trainloss_store = np.array(trainloss_store)
    mean_skill_store = np.array(mean_skill_store)
    all_skill_store = np.array(all_skill_store)
    occurence_mean_store = np.array(occurence_mean_store)

    return(trainloss_store, mean_skill_store, all_skill_store, occurence_mean_store, result3)

#------------------------------------------------------------------------------------------------------------------------------------------------------


def performances_model(trainloss_store, mean_skill_store, all_skill_store, occurence_mean_store, result3, indexes_used):
    """ plots the skills values / LRP values averaged over all the trainings"""
    mean_trainloss = trainloss_store.mean()
    std_trainloss = trainloss_store.std()
    mean_skill = mean_skill_store.mean()
    std_skill = mean_skill_store.std()
    mean_all_skill = all_skill_store[:,:,1].mean(axis=0)
    std_all_skill =all_skill_store[:,:,1].std(axis=0)
    mean_occurence = occurence_mean_store.mean(axis=0)
    std_occurence = occurence_mean_store.std(axis=0)
    print('Mean trainloss : ' + str(mean_trainloss) + ' & standard deviation : ' + str(std_trainloss))
    print('Mean skill : ' + str(mean_skill) + ' & standard deviation : ' + str(std_skill))
    print('Mean skill all years : ' + str(mean_all_skill) + ' & standard deviation : ' + str(std_all_skill))
    print('Mean LRP values/years : ' + str(mean_occurence.mean(axis=0)) + ' & standard deviation : ' + str(std_occurence.mean(axis=0)))
    
    # Create a figure and subplots
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(8,15))
    axes[0].plot(result3['year'],mean_all_skill,label='average skill')
    axes[1].bar(indexes_used, mean_occurence.mean(axis=0), label='Mean LRP values')
    axes[0].set_title('Skill per year')
    axes[0].set_xlabel('Year')
    axes[0].set_ylabel('Skill')
    axes[1].set_title('Average role in prediction of each feature')
    axes[1].set_xlabel('Features')
    axes[1].set_ylabel('Average role in prediction')
    axes[1].set_xticklabels(indexes_used, rotation='vertical')
    plt.tight_layout()
    plt.show()    


#------------------------------------------------------------------------------------------------------------------------------------------------------
#Individual MODEL
#------------------------------------------------------------------------------------------------------------------------------------------------------


class model_individual(nn.Module):
    """ Very basic model that contains one linear layer screening a (time*lat*lon, features) tensor 
    to predict a (time*lat*lon) tensor useable for SHAP values calculations"""
    def __init__(self, model_input):
        super(model_individual, self).__init__()
        self.fc1 = nn.Linear(model_input, 1)
    def forward(self, x):
        x=th.tensor(x).to(device).float()
        x = self.fc1(x)
        return(x)


#------------------------------------------------------------------------------------------------------------------------------------------------------


def trainer_individual(model, train_loader, val_loader, criterion, optimizer, num_epochs, regularization):
    """ Train function : loss is calcullated on the flattened tensor (time * lat * lon)"""
    train_losses=[]
    valid_losses=[]
    l1_lambda = 0.01
    l2_lambda = 0.01
    for epoch in range(num_epochs):
        trainloss=0
        model.train()
        tic()
        for inputs, targets in train_loader :
            inputs = inputs.to(device)
            targets = targets.unsqueeze(1).to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            outputs = outputs.to(device)
            
            if regularization == "None":
                loss = criterion(outputs, targets)
                
            if regularization == "L1":
                regularization_term = 0
                for param in model.parameters():
                    regularization_term += th.sum(th.abs(param))
                loss = criterion(outputs,targets) + l1_lambda * regularization_term
                
            if regularization == "L2":
                l2_lambda = 0.01
                regularization_term = 0
                for param in model.parameters():
                    regularization_term += th.sum(th.pow(param, 2))
                loss = criterion(outputs, targets) + l2_lambda * regularization_term

            trainloss+=loss.item()
            loss.backward(retain_graph=True)
            optimizer.step()
        train_losses.append(100*trainloss/len(train_loader))

        model.eval()  
        with th.no_grad():
            total_loss = 0.0
            for inputs, targets in val_loader:
                inputs = inputs.to(device)
                targets = targets.unsqueeze(1).to(device)
                outputs =  model(inputs)
                outputs = outputs.to(device)
                
                if regularization == "None":
                    loss = criterion(outputs, targets)
                
                if regularization == "L1":
                    regularization_term = 0
                    for param in model.parameters():
                        regularization_term += th.sum(th.abs(param))
                    loss = criterion(outputs,targets) + l1_lambda * regularization_term
                
                if regularization == "L2":
                    regularization_term = 0
                    for param in model.parameters():
                        regularization_term += th.sum(th.pow(param, 2))
                    loss = criterion(outputs, targets) + l2_lambda * regularization_term
                    
                total_loss += loss.item() 
            avg_valid_loss = 100*total_loss / len(val_loader)
        valid_losses.append(avg_valid_loss)    
        print("Number of Epoch :",format(epoch+1), "\t train loss :", format(train_losses[-1]), "\t valid loss :", format(valid_losses[-1]))
        toc()
    return(train_losses[-1],valid_losses[-1])


#------------------------------------------------------------------------------------------------------------------------------------------------------


def evaluate_individual(test_data, test_targets, target_year, time_test, value1, value2, model, datetime_objects, reversed_mask, DataLoader, batch_size, type_of_norm): 
    """ Returns predictions and targets tensors over a chosen target_year"""
    reversed_mask = reversed_mask.cpu()
    count_true = np.count_nonzero(reversed_mask)
    model.eval()
    predictions = []
    targets_list = []
    predictions_loss = []
    targets_list_loss = []
    dates_list = []
    start_date = [dt for dt in datetime_objects if dt.year == target_year]
    indexes_year = [i for i, dt in enumerate(time_test) if dt in start_date] #extract the indexes of the year of interest
    start = indexes_year[0]
    end = indexes_year[-1]+1
    indexes = [i for i in range(start*514,end*514)]
    test_dataset = TensorDataset(test_data[indexes], test_targets[indexes])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
    with th.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predicted_temperatures = model(inputs).squeeze()
            predictions.append(predicted_temperatures.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())
                
    predictions = np.concatenate(predictions)
    targets_array = np.concatenate(targets_list)
    if type_of_norm == "min_max": #opposite of normalization NORM1
        predictions = th.tensor(predictions) * (value2 - value1) + value1
        targets_array = th.tensor(targets_array) * (value2 - value1) + value1
    if type_of_norm == "mean_std": #NORM2
        predictions = th.tensor(predictions) * (value2+1e-7) + value1
        targets_array = th.tensor(targets_array) * (value2+1e-7) + value1
         
    predictions2 = predictions.view(predictions.shape[0], -1)
    targets_array2 = targets_array.view(targets_array.shape[0], -1)
    mse = mean_squared_error(targets_array2, predictions2)
    rmse = sqrt(mse)
    #reshape the denormalized predictions and targets
    predictions_reshaped = predictions.reshape(int(predictions.shape[0]/count_true),count_true)
    predictions_2D = np.zeros((predictions_reshaped.shape[0],23, 31))  # You can change the default value if needed
    predictions_2D[:,reversed_mask] = predictions_reshaped
    
    targets_reshaped = targets_array.reshape(int(targets_array.shape[0]/count_true),count_true)
    targets_2D = np.zeros((targets_reshaped.shape[0],23, 31))  # You can change the default value if needed
    targets_2D[:,reversed_mask] = targets_reshaped
    
    return(th.tensor(predictions_2D), th.tensor(targets_2D), rmse, start_date)
        
#------------------------------------------------------------------------------------------------------------------------------------------------------


def one_training_individual(reversed_mask, input_data, target_data, learning_rate, num_epochs, batch_size, datetime_objects, TensorDataset, DataLoader):
    """This is where data is split up between train, validation and test set. Preprocess the data before training """
    mask_flat = reversed_mask.flatten().to(device)
    count_true = np.count_nonzero(reversed_mask)
    ####input_data
    masked_input = input_data * reversed_mask.unsqueeze(0).unsqueeze(3)
    masked_input = masked_input.permute(0,3,1,2)
    reshaped_input = masked_input.flatten(start_dim=2).to(device)
    pertinent_input = reshaped_input[:,:,mask_flat]
    pertinent_input = pertinent_input.permute(1,0,2)
    pertinent_input = pertinent_input.flatten(start_dim=1)
    pertinent_input = pertinent_input.permute(1,0)

    #####Target data
    masked_target = target_data.to(device) * reversed_mask.unsqueeze(0).to(device)
    reshaped_target = masked_target.flatten(start_dim=1).to(device)
    pertinent_target = reshaped_target[:,mask_flat].to(device)
    pertinent_target = pertinent_target.flatten(start_dim=0).to(device)

    Ntot, Nfeature = pertinent_input.shape[0], pertinent_input.shape[1]
    validation_split = 0.2
    test_split = 0.1

    model_input = input_data.shape[3]
    
    # Determine the sizes for train, validation, and test sets based on the time dimension
    time_steps = input_data.shape[0]
    train_size = int(0.75 * time_steps) * count_true
    val_size = int(0.05 * time_steps) * count_true
    test_size = time_steps* count_true - train_size - val_size

    # Split the dataset based on the time dimension
    train_data = pertinent_input[:train_size, :].to(device)
    train_targets = pertinent_target[:train_size].to(device)
    val_data = pertinent_input[train_size:train_size+val_size, :].to(device)
    val_targets = pertinent_target[train_size:train_size+val_size].to(device)
    test_data = pertinent_input[-test_size:, :].to(device)
    test_targets = pertinent_target[-test_size:].to(device)
    time_test = datetime_objects[-int(test_size/count_true):]

    train_dataset = TensorDataset(train_data, train_targets)
    val_dataset = TensorDataset(val_data, val_targets)
    test_dataset = TensorDataset(test_data, test_targets)

    # Create data loaders for train, validation, and test sets
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    return(train_loader, val_loader, test_loader, time_test, model_input, test_data, test_targets)


#------------------------------------------------------------------------------------------------------------------------------------------------------


def SHAP_map(test_data, test_targets, target_year, time_test, type_of_norm, reversed_mask, datetime_objects, DataLoader, batch_size, model, value1, value2):
    """ Creates a tensor of size (time, lat, lon, shap) that stores the SHAP values of every grid point over one chosen target year"""
    count_true = np.count_nonzero(reversed_mask)
    model.eval()
    inputs_list = []
    predictions = []
    targets_list = []
    predictions_loss = []
    targets_list_loss = []
    dates_list = []

    start_date = [dt for dt in datetime_objects if dt.year == target_year]
    indexes_year = [i for i, dt in enumerate(time_test) if dt in start_date] #extract the indexes of the year of interest
    start = indexes_year[0]
    end = indexes_year[-1]+1
    indexes = [i for i in range(start*count_true,end*count_true)]
    test_dataset = TensorDataset(test_data[indexes], test_targets[indexes])
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle = False)
    
    model.eval()
    with th.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.to(device)
            targets = targets.to(device)
            predicted_temperatures = model(inputs).squeeze()
            inputs_list.append(inputs.detach().cpu().numpy())
            predictions.append(predicted_temperatures.detach().cpu().numpy())
            targets_list.append(targets.detach().cpu().numpy())
                
    inputs_list = np.concatenate(inputs_list)
    predictions = np.concatenate(predictions)
    targets_array = np.concatenate(targets_list)
    
    def forward_fn(x):
        x= th.tensor(x).to(device)
        return model(x)
    
    # Initialize the SHAP explainer with your model's forward function
    explainer = shap.Explainer(forward_fn, masker=shap.maskers.Independent(inputs_list))

    # Calculate the SHAP values
    shap_values = explainer.shap_values(inputs_list)

    #print(targets_array.shape)
    if type_of_norm == "min_max": #opposite of normalization NORM1
        predictions = th.tensor(predictions) * (value2 - value1) + value1
        targets_array = th.tensor(targets_array) * (value2 - value1) + value1
    if type_of_norm == "mean_std": #NORM2
        predictions = th.tensor(predictions) * (value2+1e-7) + value1
        targets_array = th.tensor(targets_array) * (value2+1e-7) + value1
     
    predictions2 = predictions.view(predictions.shape[0], -1)
    targets_array2 = targets_array.view(targets_array.shape[0], -1)
    mse = mean_squared_error(targets_array2, predictions2)
    rmse = sqrt(mse)
    
    #reshape the denormalized predictions and targets
    predictions_reshaped = predictions.reshape(int(predictions.shape[0]/count_true),count_true)
    predictions_2D = np.zeros((predictions_reshaped.shape[0],23, 31))  # You can change the default value if needed
    predictions_2D[:,reversed_mask] = predictions_reshaped

    targets_reshaped = targets_array.reshape(int(targets_array.shape[0]/count_true),count_true)
    targets_2D = np.zeros((targets_reshaped.shape[0],23, 31))  # You can change the default value if needed
    targets_2D[:,reversed_mask] = targets_reshaped
    
    #reshape the shap values
    shap_values_reshaped = shap_values.reshape(int(shap_values.shape[0]/count_true),count_true,inputs_list.shape[1])
    shap_values_tensor = np.zeros((shap_values_reshaped.shape[0], 23, 31, shap_values_reshaped.shape[2]))
    shap_values_tensor[:, reversed_mask, :]= shap_values_reshaped
    return(th.tensor(predictions_2D), th.tensor(targets_2D), rmse, start_date, shap_values, shap_values_tensor)
        
    
#------------------------------------------------------------------------------------------------------------------------------------------------------


def get_skill_fn_individual(test_data, test_targets, time_test, is_mask, reversed_mask, type_of_norm, model, datetime_objects, DataLoader, batch_size, value1, value2):
    """ Calculates the skills for all the years betwen 2011 ad 2017 to evaluate the model's performances"""
    reversed_mask = reversed_mask.to(device)
    years=[2011+i for i in range(7)]
    result3=pd.DataFrame()
    result3['year']=years
    skills=[]
    for year in years:
        predictions, targets_array, mse, start_date = evaluate_individual(test_data, test_targets, year, time_test, value1, value2, model, datetime_objects, reversed_mask, DataLoader, batch_size, type_of_norm)
        if is_mask == False:
            reshaped_predictions = predictions.reshape(predictions.shape[0],predictions.shape[1]*predictions.shape[2]).to(device)
            reshaped_targets = targets_array.reshape(targets_array.shape[0],targets_array.shape[1]*targets_array.shape[2]).to(device)
            pertinent_predictions = reshaped_predictions.permute(1,0)
            pertinent_targets = reshaped_targets.permute(1,0)
        elif is_mask == True:   
            masked_predictions = predictions.to(device) * reversed_mask
            masked_targets = targets_array.to(device) * reversed_mask
            reshaped_predictions = masked_predictions.flatten(start_dim=1).to(device)
            reshaped_targets = masked_targets.flatten(start_dim=1).to(device)
            mask_gpu = reversed_mask.flatten().to(device)
            pertinent_predictions = reshaped_predictions[:,mask_gpu]
            pertinent_targets = reshaped_targets[:,mask_gpu]
            pertinent_predictions = pertinent_predictions.permute(1,0)
            pertinent_targets = pertinent_targets.permute(1,0)
    
        flattened_predictions = pertinent_predictions.flatten()
        flattened_targets = pertinent_targets.flatten()
    
        numpy_array_flat_pred = flattened_predictions.cpu().numpy()
        numpy_array_flat_targ = flattened_targets.cpu().numpy()
    
        df_flat_pred = pd.DataFrame(numpy_array_flat_pred)
        df_flat_targ = pd.DataFrame(numpy_array_flat_targ)
    
        concatenated_dates = start_date * pertinent_targets.shape[0] #514 = number of interesting points

        start_date = pd.DataFrame(concatenated_dates)
        results = pd.concat((start_date, df_flat_targ, df_flat_pred),axis=1)
        results.columns = ['start_date', 'targets_array', 'predictions']
        skill=get_col_skill(results,'targets_array','predictions',date_col='start_date')
        skills.append(skill)
    result3['skill']=skills
    return(result3, mse)


#------------------------------------------------------------------------------------------------------------------------------------------------------


def visualization_skills_individual(test_data,test_targets,time_test,is_mask, reversed_mask, type_of_norm,model, datetime_objects, DataLoader, batch_size, value1, value2):
    """ Plots the  skills over the years 2011-2017"""
    result3, mse = get_skill_fn_individual(test_data,test_targets,time_test,is_mask, reversed_mask, type_of_norm,model, datetime_objects, DataLoader, batch_size, value1, value2)
    results = np.array(result3['skill'])
    print(result3)
    print(results.mean())
    plt.plot(result3['year'],result3['skill'],label='NN')
    plt.legend()
    plt.xlabel('Year')
    plt.ylabel('Skill')
    plt.show()
