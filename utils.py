import torch
import time
import copy
from torch import nn
from torch.optim import lr_scheduler, Adam
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import mean_squared_error

def train_model_snapshot(model, criterion, metric, lr, dataloaders, dataset_sizes, device, num_cycles, num_epochs_per_cycle, n_extra):
    since = time.time()
    
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1000000.0
    model_w_arr = []
    optimizer = Adam(model.parameters(), lr=lr)
    for cycle in range(num_cycles):
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs_per_cycle)
        for epoch in range(num_epochs_per_cycle):
            print('Cycle {}: Epoch {}/{}'.format(cycle, epoch, num_epochs_per_cycle - 1))
            print('-' * 10)
            
            # Each epoch has a training and validation phase
            for phase in ['train', 'val']:
                if phase == 'train':
                    model.train()  # Set model to training mode
                else:
                    model.eval()   # Set model to evaluate mode
                
                running_loss = np.zeros((3,))
                
                # Iterate over data.
                for inputs, targets, masks, idx, errors, snr in dataloaders[phase]:
                    inputs = inputs.to(device)
                    targets = targets.to(device)
                    masks = masks.to(device)
                    idx = idx.to(device)
                    errors = errors.to(device)
                    snr = snr.to(device)
                    targets = targets.reshape(-1, 5)
                    errors = errors.reshape(-1, 5)
                    
                    # zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # forward
                    with torch.set_grad_enabled(phase == 'train'):
                        outputs = model(inputs, masks, idx)
                        loss = criterion(outputs, targets, errors, snr)
                        metric_loss = metric(outputs[:,:3], targets[:,:3])
                        # backward + optimize only if in training phase
                        if phase == 'train':
                            loss.backward()
                            optimizer.step()
                    
                    # statistics
                    running_loss += metric_loss.sum(0).detach().cpu().numpy()
                
                epoch_loss = np.sqrt(running_loss / (dataset_sizes[phase]*68)).mean()
                if phase == 'val':
                    scheduler.step()#epoch_loss)
                
                print('{} Loss: {:.4f}'.format(
                    phase, epoch_loss))
                
                # deep copy the model
                if phase == 'val' and epoch_loss < best_loss:
                    best_loss = epoch_loss
            print()
        # deep copy snapshot
        model_w_arr.append(copy.deepcopy(model.state_dict()))
        
    ensemble_loss = np.zeros((3,))
    
    val_pred = torch.zeros((dataset_sizes['val']*68, 3)).float()
    val_target = torch.zeros((dataset_sizes['val']*68, 3)).float()
    #predict on validation using snapshots
    i = 0
    for inputs, targets, masks, idx,  _, _ in dataloaders['val']:
        inputs = inputs.to(device)
        targets = targets.to(device)
        masks = masks.to(device)
        idx = idx.to(device)
        targets = targets.reshape(-1, 5)
        
        # forward
        for weights in model_w_arr:
            model.load_state_dict(weights)
            model.eval()
            val_pred[i:i+inputs.shape[0]*68] += model(inputs, masks, idx).detach().cpu()[:,:3] / len(model_w_arr)
        
        val_target[i:i+inputs.shape[0]*68] = targets.detach().cpu()[:,:3]
        
        i += inputs.shape[0]*68
    
    #test time augmentation averaging
    n = val_pred.shape[0]//n_extra
    
    avg_val_pred = torch.zeros((n, 3)).float()
    avg_target_pred = torch.zeros((n, 3)).float()
    for j in range(0, n, 68):
        k = j*n_extra
        for sh in range(n_extra):
            avg_val_pred[j:j+68] += val_pred[k + 68*sh: k + 68*(sh+1)]/n_extra
            avg_target_pred[j:j+68] += val_target[k + 68*sh: k + 68*(sh+1)]/n_extra
    
    val_pred = avg_val_pred
    val_target = avg_target_pred
    ensemble_loss = torch.sqrt(metric(val_pred, val_target).mean(0)).mean()
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Ensemble Loss : {:4f}, Best val Loss: {:4f}'.format(ensemble_loss, best_loss))
    
    # load snapshot model weights and combine them in array
    model_arr =[]
    for weights in model_w_arr:
        model.load_state_dict(weights)   
        model_arr.append(model) 
    
    val_pred = val_pred.numpy()
    val_target = val_target.numpy()
    
    return model_arr, ensemble_loss, val_pred, val_target

def pretrain_model(model, lr, dataloaders, dataset_sizes, device, num_epochs=25):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    optimizer = Adam(model.parameters(), lr=lr)
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss(reduce = 'none')

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, idx, labels, mask in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)
                idx = idx.to(device)
                mask = mask.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model.forward_pretrain(inputs, idx)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    loss = (loss * mask).sum()/mask.sum()

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds[:, :107] == labels.data[:, :107])
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / (dataset_sizes[phase]*107)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def test(models_arr, loader, device, n_samples, offset, n_extra):
    res = np.zeros((n_samples*5, 3), dtype = np.float32)
    for model in models_arr:
        model.eval()
        res_arr = []
        for inputs, masks, idx in loader:
            inputs = inputs.to(device)
            masks = masks.to(device)
            idx = idx.to(device)
            # forward
            with torch.set_grad_enabled(False):
                outputs = model(inputs, masks, idx)[:,:3]
                res_arr.append(outputs.detach().cpu().numpy())
        res_arr = np.concatenate(res_arr, axis = 0)
        res += res_arr / len(models_arr)
        
    res_tta = np.zeros((n_samples, 3), dtype = np.float32)
    for j in range(0, n_samples, offset):
        k = j*n_extra
        for sh in range(n_extra):
            res_tta[j:j+offset] += res[k + offset*sh: k + offset*(sh+1)]/n_extra
    return res_tta 

def MCRMSE(y_true, y_pred):
    colwise_mse = torch.mean(torch.square(y_true - y_pred), dim=1)
    return torch.mean(torch.sqrt(colwise_mse), dim=1)


def sn_mcrmse_loss_v2(predict, target, error, signal_to_noise):
    predict = predict.reshape(-1, 68, 5)
    target = target.reshape(-1, 68, 5)
    loss = MCRMSE(target, predict)
    weight = 0.5 * torch.log(signal_to_noise + 1.01)
    loss = (loss * weight).mean()
    return loss

def snr_mcrmse_loss(predict, target, error, signal_to_noise):
    signal_to_noise = torch.log(signal_to_noise+1.1)/2
    batch_size = target.shape[0]//68
    ###signal_to_noise = signal_to_noise+0.1 #reshape snr?
    eps = F.relu(error - 0.25)  #svm tube loss
    l = torch.abs(predict - target)
    l = F.relu(l - eps)

    l = l ** 2
    l = l.reshape(batch_size, 68, -1) * signal_to_noise.reshape(batch_size, 1, 1)
    l = l.reshape(batch_size * 68, -1)
    l = l.sum(0) / signal_to_noise.sum().item()
    l = l ** 0.5
    loss = l.mean()
    return loss

def comp_metric(pred, target):
    return np.mean([np.sqrt(mean_squared_error(pred[:,c], target[:,c])) for c in range(3)])
