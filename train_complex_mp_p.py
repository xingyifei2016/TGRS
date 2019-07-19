# -*- coding: utf-8 -*-
import model_p_mod as model

import torch
import time
import numpy as np
import torch.optim as optim
import torch.nn as nn
from torch.utils import data
import data_utils
import random
import math
import matplotlib.pyplot as plt
import scipy.signal
from scipy.interpolate import make_interp_spline, BSpline
import scipy.interpolate as interpolate
from logger import setup_logger
import os
import random
from os import listdir
from os.path import isfile, join
from torch.optim.lr_scheduler import ReduceLROnPlateau
import pandas as pd

# Parameters for data loading
params_train = {
          'shuffle': True,
          'num_workers': 1}

params_val = {'batch_size': 400,
          'shuffle': True,
          'num_workers': 1}

max_epochs = 100

nC1 = 1285
nC2 = 429
nC3 = 6694
nC4 = 451
nC5 = 1164
nC6 = 1415
nC7 = 573
nC8 = 572
nC9 = 573
nC10 = 1401
nC11 = 1159

def data_prep_10(batch_size, split=0.7):
    data_x = []
    data_y = []
    for f in listdir('/data_polar'):
        data = np.load(join('/data_polar', f))
        label = f.split('_')[0].split('c')[1]
        if int(label) != 11:
            data_x.append(data)
            data_y.append(int(label)-1)

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    xshape = data_x.shape
    
    data_x = data_x.reshape((xshape[0], xshape[1], 1, xshape[2], xshape[3]))
    
    data_x[:, 0,...] = np.arccos(data_x[:, 0, 0,...]).reshape(data_x[:, 0,...].shape)
    data_x[:, 1,...] = data_x[:, 4,...]
    data_x = data_x[:, :2,...]
    
        
    
    
    
    split = int(14557*split)
#     x_train = data_x[int(len(data_y)*split):]
#     y_train = data_y[int(len(data_y)*split):]

#     x_test = data_x[:int(len(data_y)*split)]
#     y_test = data_y[:int(len(data_y)*split)]
    
    data_set_11 = torch.utils.data.TensorDataset(torch.from_numpy(data_x).type(torch.FloatTensor), torch.from_numpy (data_y).type(torch.LongTensor))
    
    idx = np.random.permutation(14557)
    test_idx = idx[:split]
    train_idx = idx[split:]   
    
    data_train = torch.utils.data.Subset(data_set_11,indices=train_idx)
    data_test = torch.utils.data.Subset(data_set_11,indices=test_idx)
    
    params_train = {
          'shuffle': True,
          'num_workers': 1}

    params_val = {'batch_size': 400,
              'shuffle': True,
              'num_workers': 1}
    
    train_generator = torch.utils.data.DataLoader(dataset=data_train, batch_size = batch_size, **params_train)
    test_generator = torch.utils.data.DataLoader(dataset=data_test, **params_val)
    
    
#     train_loader = torch.utils.data.TensorDataset(torch.from_numpy(x_train).type(torch.FloatTensor), torch.from_numpy (y_train).type(torch.LongTensor))
#     train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = True)
#     test_loader = torch.utils.data.TensorDataset(torch.from_numpy(x_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor))
#     test_loader_dataset = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle = False)
    
    return train_generator, test_generator #train_loader_dataset, test_loader_dataset



def data_prep_11(batch_size, split=0.7):
    data_x = []
    data_y = []
    for f in listdir('../data_polar'):
        data = np.load(join('../data_polar', f))
        label = f.split('_')[0].split('c')[1]
#         if int(label) != 11:
        data_x.append(data)
        data_y.append(int(label)-1)

    data_x = np.array(data_x)
    data_y = np.array(data_y)
    
    xshape = data_x.shape
    
    data_x = data_x.reshape((xshape[0], xshape[1], 1, xshape[2], xshape[3]))
    
    data_x[:, 0,...] = np.arccos(data_x[:, 0, 0,...]).reshape(data_x[:, 0,...].shape)
    data_x[:, 1,...] = data_x[:, 4,...]
    data_x = data_x[:, :2,...]
    
        
    
    
    
    
#     x_train = data_x[int(len(data_y)*split):]
#     y_train = data_y[int(len(data_y)*split):]

#     x_test = data_x[:int(len(data_y)*split)]
#     y_test = data_y[:int(len(data_y)*split)]
    
    data_set_11 = torch.utils.data.TensorDataset(torch.from_numpy(data_x).type(torch.FloatTensor), torch.from_numpy (data_y).type(torch.LongTensor))
    train_idx, test_idx = index_split(False)
    data_train = torch.utils.data.Subset(data_set_11,indices=train_idx)
    data_test = torch.utils.data.Subset(data_set_11,indices=test_idx)
    
    params_train = {
          'shuffle': True,
          'num_workers': 1}

    params_val = {'batch_size': 400,
              'shuffle': True,
              'num_workers': 1}
    
    train_generator = torch.utils.data.DataLoader(dataset=data_train, batch_size = batch_size, **params_train)
    test_generator = torch.utils.data.DataLoader(dataset=data_test, **params_val)
    
    
#     train_loader = torch.utils.data.TensorDataset(torch.from_numpy(x_train).type(torch.FloatTensor), torch.from_numpy (y_train).type(torch.LongTensor))
#     train_loader_dataset = torch.utils.data.DataLoader(train_loader, batch_size=batch_size, shuffle = True)
#     test_loader = torch.utils.data.TensorDataset(torch.from_numpy(x_test).type(torch.FloatTensor), torch.from_numpy(y_test).type(torch.LongTensor))
#     test_loader_dataset = torch.utils.data.DataLoader(test_loader, batch_size=batch_size, shuffle = False)
    
    return train_generator, test_generator #train_loader_dataset, test_loader_dataset


#Default device
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
logger = setup_logger("Comparison")


def index_split(full_or_no):
    csv_path = 'chipinfo.csv' #os.path.join('./save/', 'split[{split}]-10class-model-temp.ckpt'.format(split=i))
    df = pd.read_csv(csv_path)

    training = df.loc[df['depression'] == 17]

    subclass_9 = training.loc[training['target_type'] != 'bmp2_tank']
    subclass_8 = subclass_9.loc[subclass_9['target_type'] != 't72_tank'].index.values


    class_1_train = np.array(training.loc[training['serial_num']=='c21'].index.values)
    class_3_train = np.array(training.loc[training['serial_num']=='132'].index.values)


    subclass = np.concatenate([subclass_8, class_1_train, class_3_train], axis=0)
    training = training.index

    testing = df.loc[df['depression'] == 15]

    subclass_test9 = testing.loc[testing['target_type'] != 'bmp2_tank']
    subclass_test8 = np.array(subclass_test9.loc[subclass_test9['target_type']=='t72_tank'].index.values)

    class_1_test1 = np.array(testing.loc[testing['serial_num']=='c21'].index.values)
    class_1_test2 = np.array(testing.loc[testing['serial_num']=='9563'].index.values)
    class_1_test3 = np.array(testing.loc[testing['serial_num']=='9566'].index.values)

    class_3_test1 = np.array(testing.loc[testing['serial_num']=='132'].index.values)
    class_3_test2 = np.array(testing.loc[testing['serial_num']=='812'].index.values)
    class_3_test3 = np.array(testing.loc[testing['serial_num']=='s7'].index.values)

    subclass_test = np.concatenate([subclass_test8, class_1_test1, class_1_test2, class_1_test3, class_3_test1, class_3_test2, class_3_test3], axis=0)
    
    testing = np.array(testing.index.values)
        
    if full_or_no:
        return training, testing
    else:
        return subclass, subclass_test
        
    
    
    
    
    
    
def test(model, device, test_loader):
    
    test_loss = 0
    correct = 0
    res = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            
            
            ####
            data[:, 1,...]=torch.zeros(data[:, 1,...].shape)
            ####
            
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #For confusion matrix
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return 100. * correct / len(test_loader.dataset)



# f= open("10class-results-emp.txt","w+")
 

# # # #10 class
# accs = []
# batches = [30, 50, 80, 100, 200, 500]
# lr = [0.05, 0.03, 0.01, 0.008]
# splitting = [0.5, 0.6, 0.7, 0.8, 0.9]

# for b in batches:
#     for lr in lr:
#         accs = []
#         for s in splitting:
            
#             logger.info('Batch[{b}]-LR[{lr}]-Split[{sp}]'.format(b=b, lr=lr, sp=s))
            
#             manifold_net = model.ManifoldNetComplex().cuda()
#             init_params = manifold_net.parameters()
#             model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
#             params = sum([np.prod(p.size()) for p in model_parameters])
#             logger.info("Model Parameters: "+str(params))
            
#             optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
#             criterion = nn.CrossEntropyLoss()

#             def classification(out,desired):
#                 _, predicted = torch.max(out, 1)
#                 total = desired.shape[0]
#                 correct = (predicted == desired).sum().item()
#                 return correct

#             # Training...
#             print('Starting training...')
#             validation_accuracy = []
#             highest=0
#             try:
#                 #for train_idx, test_idx in data_utils.k_folds(n_splits=10, n_samples=(14557)):
                    
#                     np.random.seed(42222222)
                    

#                     train_generator, test_generator = data_prep_10(b, s)
#                     epoch_validation_history = []
#                     for epoch in range(max_epochs):
#                         print('Starting Epoch ', epoch, '...')
#                         loss_sum = 0
#                         start = time.time()
#                         train_acc = 0
#                         for it,(local_batch, local_labels) in enumerate(train_generator):
#                             batch = torch.tensor(local_batch, requires_grad=True).cuda()
#                             optimizer.zero_grad()
#                             out = manifold_net(batch)
#                             train_acc += classification(out, local_labels.cuda()) 

#                             loss = criterion(out, local_labels.cuda())
#                             loss.backward()
#                             optimizer.step()
#                         logger.info("Epoch: "+str(epoch)+"Training accuracy: "+str(train_acc / len(train_generator.dataset)*100.))

#                         acc = test(manifold_net, device, test_generator)
#                         if acc > highest:
#                             highest=acc
#                             save_path = os.path.join('./save/', 'split[{split}]-batch[{batch}]-lr[{lr}]-10class-model-norm.ckpt'.format(split=s, batch=b, lr=lr))
#                             torch.save(manifold_net.state_dict(), save_path)
#                             print('Saved model checkpoints into {}...'.format(save_path))
#                         logger.info("Epoch: "+str(epoch)+"Testing accuracy is "+str(acc))
#                         end = time.time()
#                         print('Epoch Time:', end-start)
#     #                 accs.append(highest)
#                     manifold_net = model.ManifoldNetComplex().cuda()
#                     optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
#                     criterion = nn.CrossEntropyLoss()
#                     accs.append(highest)
#             except KeyboardInterrupt:
#                 pass
#         f.write(str(accs) + " - batch[{b}] - splitting [{sp}] - lr [{lr}]".format(b=b, lr=lr, sp=s))
# f.close()
        
        
        


# for b in batches:
#     for lr in lr:
#         logger.info('Batch[{b}]-LR[{lr}]'.format(b=b, lr=lr))
#         manifold_net = model.ManifoldNetComplex().cuda()


#         init_params = manifold_net.parameters()
#         model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         logger.info("Model Parameters: "+str(params))
#         #manifold_net.load_state_dict(torch.load('models/pretrained.pt'))
#         optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
#         criterion = nn.CrossEntropyLoss()

#         def classification(out,desired):
#             _, predicted = torch.max(out, 1)
#             total = desired.shape[0]
#             correct = (predicted == desired).sum().item()
#             return correct

#         # Training...
#         print('Starting training...')
#         validation_accuracy = []
#         highest=0
#         #split = 0
#         try:
#             #for train_idx, test_idx in data_utils.k_folds(n_splits=10, n_samples=(14557)):
#                 np.random.seed(42222222)
#                 train_idx, test_idx = index_split(False)
#                 data_train = data.Subset(data_set_10,indices=train_idx)
#                 data_test = data.Subset(data_set_10,indices=test_idx)

#                 train_generator = torch.utils.data.DataLoader(dataset=data_train, batch_size = b, **params_train)
#                 test_generator = torch.utils.data.DataLoader(dataset=data_test, **params_val)
#                 epoch_validation_history = []
#                 for epoch in range(max_epochs):
#                     print('Starting Epoch ', epoch, '...')
#                     loss_sum = 0
#                     start = time.time()
#                     train_acc = 0
#                     for it,(local_batch, local_labels) in enumerate(train_generator):
#                         batch = torch.tensor(local_batch, requires_grad=True).cuda()
#                         optimizer.zero_grad()
#                         out = manifold_net(batch)
#                         train_acc += classification(out, local_labels.cuda()) 

#                         loss = criterion(out, local_labels.cuda())
#                         loss.backward()
#                         optimizer.step()
#                     logger.info("Epoch: "+str(epoch)+"Training accuracy: "+str(train_acc / len(train_generator.dataset)*100.))

#                     acc = test(manifold_net, device, test_generator)
#                     if acc > highest:
#                         highest=acc
#                         save_path = os.path.join('./save/', 'split[{split}]-batch[{batch}]-lr[{lr}]-10class-model-norm.ckpt'.format(split=i, batch=b, lr=r))
#                         torch.save(manifold_net.state_dict(), save_path)
#                         print('Saved model checkpoints into {}...'.format(save_path))
#                     logger.info("Epoch: "+str(epoch)+"Testing accuracy is "+str(acc))
#                     end = time.time()
#                     print('Epoch Time:', end-start)
# #                 accs.append(highest)
#                 manifold_net = model.ManifoldNetComplex().cuda()
#                 optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
#                 criterion = nn.CrossEntropyLoss()
#         except KeyboardInterrupt:
#             pass


# splitting=np.array(splitting)
# accs = np.array(accs)

# t, c, k = interpolate.splrep(splitting, accs, s=0, k=4)
# xmin, xmax = splitting.min(), splitting.max()
# xx = np.linspace(xmin, xmax, 100)
# spline = interpolate.BSpline(t, c, k, extrapolate=False)

# fig, ax = plt.subplots()

# # ax.plot(splitting, accs, 'bo', label='Original points')
# ax.plot(xx, spline(xx))
# ax.grid()
# ax.legend(loc='best')


# ax.set_ylabel('Testing Accuracy')
# ax.set_xlabel('Testing Data Percentage')
# fig.savefig('10class-testing_acc-temp.png', dpi=fig.dpi)
# f= open("10class-results-emp.txt","w+")
# f.write("splitting percentage: ")
# f.write(str(splitting))
# f.write("accuracy is: ")
# f.write(str(accs))
# f.close() 


def calc_next_size(wh, kern, stride):
    return int(math.floor((wh - kern) / stride + 1))


def generate_random_complex_regular_array(input_size):
    
    complex_c, complex_kern, complex_stride, regular_c, regular_kern, regular_stride, max_pool = [], [], [], [], [], [], []
    middle = 0
    limit_complex = 2 #Limit number of complex convolutions to 2
    
    num_complex_conv = np.random.randint(2, 4)
    prev_comp = 1
    
    for i in range(num_complex_conv):
        kern = np.random.randint(3, 8)
        stride = np.random.randint(1, 4)
        while calc_next_size(input_size, kern, stride) + i < 17:
            kern = np.random.randint(3, 8)
            stride = np.random.randint(1, 4)
        new_comp = np.random.randint(0, prev_comp+1)
        while prev_comp + new_comp - i > 5:
            new_comp = np.random.randint(0, prev_comp+1)
        prev_comp = prev_comp + new_comp
        complex_c.append(prev_comp*10)
        complex_kern.append(kern)
        complex_stride.append(stride)
        input_size = calc_next_size(input_size, kern, stride)
        if i == num_complex_conv-1:
            middle = input_size * input_size
        
    while input_size != 1:
        kern = np.random.randint(3, 8)
        stride = np.random.randint(1, 4)
        
        if calc_next_size(input_size, kern, stride) == 1:
            new_comp = np.random.randint(0, prev_comp+1)
            while prev_comp + new_comp - i > 7:
                new_comp = np.random.randint(0, prev_comp+1)
            prev_comp = prev_comp + new_comp
            regular_c.append(prev_comp*10)
            regular_kern.append(kern)
            regular_stride.append(stride)
            max_pool.append(1)
            return complex_c, complex_kern, complex_stride, regular_c, regular_kern, regular_stride, middle, max_pool
            
        elif calc_next_size(input_size, kern, stride) > 1:
            regular_kern.append(kern)
            regular_stride.append(stride)
            new_comp = np.random.randint(0, prev_comp+1)
            while prev_comp + new_comp - i > 7:
                new_comp = np.random.randint(0, prev_comp+1)
            prev_comp = prev_comp + new_comp
            regular_c.append(prev_comp*10)
            input_size = calc_next_size(input_size, kern, stride)
            
            pool = np.random.randint(1, 4)
            
            if pool == 0:
                pool = 1
                
            while calc_next_size(input_size, pool, pool) < 1:
                pool = np.random.randint(1, 4)
                if calc_next_size(input_size, pool, pool) == 1:
                    max_pool.append(pool)
                    return complex_c, complex_kern, complex_stride, regular_c, regular_kern, regular_stride, middle, max_pool
                
            max_pool.append(pool)
            input_size = calc_next_size(input_size, pool, pool)
    return complex_c, complex_kern, complex_stride, regular_c, regular_kern, regular_stride, middle, max_pool
            
            
        
            
                
            
# save_path = None
# train_generator, test_generator = data_prep_11(150, 0.3)
# highest_acc = 0
# while True:
#     params = {}
#     try:
#         complex_c, complex_kern, complex_stride, regular_c, regular_kern, regular_stride, middle, max_pool = generate_random_complex_regular_array(100)
#         logger.info("Complex_Channels[{cc}]-Complex_Kern[{ck}]-Complex_Stride[{cs}]".format(cc=str(complex_c), ck=str(complex_kern), cs=str(complex_stride)))
#         logger.info("Regular_Channels[{cc}]-Regular_Kern[{ck}]-Regular_Stride[{cs}]-Max_Pool[{mp}]".format(cc=str(regular_c), ck=str(regular_kern), cs=str(regular_stride), mp=str(max_pool)))
#         params={"complex_c": complex_c, "complex_kern": complex_kern, "complex_stride": complex_stride, "regular_c": regular_c, "regular_kern": regular_kern, "regular_stride": regular_stride, "middle": middle, "max_pool": max_pool}
        
#         manifold_net = model.CustomManifold(params).cuda()
#         init_params = manifold_net.parameters()
#         model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
#         params = sum([np.prod(p.size()) for p in model_parameters])
#         if params > 150000:
#             continue
#         logger.info("Model Parameters: "+str(params))
#         logger.info(str(manifold_net))
#         #manifold_net.load_state_dict(torch.load('models/pretrained.pt'))
#         optimizer = optim.Adam(manifold_net.parameters(), lr=0.02)
#         criterion = nn.CrossEntropyLoss()

#         def classification(out,desired):
#             _, predicted = torch.max(out, 1)
#             total = desired.shape[0]
#             correct = (predicted == desired).sum().item()
#             return correct

#         # Training...
#         print('Starting training...')
        
#         np.random.seed(42222222)
        
#         epoch_validation_history = []
        
#         for epoch in range(max_epochs):
#             print('Starting Epoch ', epoch, '...')
#             loss_sum = 0
#             start = time.time()
#             train_acc = 0
#             for it,(local_batch, local_labels) in enumerate(train_generator):

#                 batch = torch.tensor(local_batch, requires_grad=True).cuda()

#                 optimizer.zero_grad()
# #                         print(batch.shape)
#                 out = manifold_net(batch)
#                 train_acc += classification(out, local_labels.cuda()) 

#                 loss = criterion(out, local_labels.cuda())
#                 loss.backward()
#                 optimizer.step()
#             logger.info("Epoch: "+str(epoch)+" - Training accuracy: "+str(train_acc / len(train_generator.dataset)*100.))

#             acc = test(manifold_net, device, test_generator)
#             if acc > highest_acc:
#                 highest_acc =acc
#                 if save_path != None:
#                     os.remove(save_path)
#                 save_path = os.path.join('./save/', 'acc[{acc}]-11class-model.ckpt'.format(acc = np.round(acc, 3)))
#                 torch.save(manifold_net.state_dict(), save_path)
#                 print('Saved model checkpoints into {}...'.format(save_path))
#             logger.info("Epoch: "+str(epoch)+" - Testing accuracy is "+str(acc))
#             end = time.time()
#             print('Epoch Time:', end-start)
# #                 accs.append(highest)
#     except:
#         pass
    
        
                            
                    
                            
            
            
    












#11 class
accs = []
batches = [150]
lrs = [0.02]


for b in batches:
    for lr in lrs:
        logger.info('Batch[{b}]-LR[{lr}]'.format(b=b, lr=lr))
        manifold_net = model.ManifoldNetAll().cuda()
        init_params = manifold_net.parameters()
        model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        logger.info("Model Parameters: "+str(params))
        #manifold_net.load_state_dict(torch.load('models/pretrained.pt'))
        optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        def classification(out,desired):
            _, predicted = torch.max(out, 1)
            total = desired.shape[0]
            correct = (predicted == desired).sum().item()
            return correct

        # Training...
        print('Starting training...')
        validation_accuracy = []
        highest=0
        #split = 0
        try:
            #for train_idx, test_idx in data_utils.k_folds(n_splits=10, n_samples=(14557)):
                np.random.seed(42222222)
                train_generator, test_generator = data_prep_11(b, 0.3)
                epoch_validation_history = []
                save_path = None
                for epoch in range(max_epochs):
                    print('Starting Epoch ', epoch, '...')
                    loss_sum = 0
                    start = time.time()
                    train_acc = 0
                    for it,(local_batch, local_labels) in enumerate(train_generator):
                        
                        batch = torch.tensor(local_batch, requires_grad=True).cuda()
                        
                        
                        ####
                        batch[:, 1,...] = torch.zeros(batch[:, 1,...].shape)
                        ####
                        
                        
                        
                        
                        optimizer.zero_grad()
#                         print(batch.shape)
                        out = manifold_net(batch)
                        train_acc += classification(out, local_labels.cuda()) 

                        loss = criterion(out, local_labels.cuda())
                        loss.backward()
                        optimizer.step()
                    logger.info("Epoch: "+str(epoch)+" - Training accuracy: "+str(train_acc / len(train_generator.dataset)*100.))

                    acc = test(manifold_net, device, test_generator)
                    if acc > highest:
                        highest=acc
                        if save_path != None:
                            os.remove(save_path)
                        save_path = os.path.join('./save/', 'batch[{batch}]-lr[{lr}]-acc[{acc}]-11class-model-1s.ckpt'.format(batch=b, lr=lr, acc = np.round(acc, 3)))
                        torch.save(manifold_net.state_dict(), save_path)
                        print('Saved model checkpoints into {}...'.format(save_path))
                    logger.info("Epoch: "+str(epoch)+" - Testing accuracy is "+str(acc))
                    end = time.time()
                    print('Epoch Time:', end-start)
#                 accs.append(highest)
                manifold_net = model.ManifoldNetAll().cuda()
                optimizer = optim.Adam(manifold_net.parameters(), lr=lr)
                criterion = nn.CrossEntropyLoss()
        except KeyboardInterrupt:
            pass

logger.info('random stuff')
while True:
    for it,(local_batch, local_labels) in enumerate(train_generator):
        batch = torch.tensor(local_batch, requires_grad=True).cuda()
        optimizer.zero_grad()
        out = manifold_net(batch)
        train_acc += classification(out, local_labels.cuda()) 

        loss = criterion(out, local_labels.cuda())
        loss.backward()
        optimizer.step()
    logger.info("Epoch: "+str(epoch)+"Training accuracy: "+str(train_acc / len(train_generator.dataset)*100.))

    acc = test(manifold_net, device, test_generator)
#     scheduler.step(acc)
    if acc > highest:
        highest=acc
        save_path = os.path.join('./save/', 'split[{split}]-11class-model.ckpt'.format(split=i))
        torch.save(manifold_net.state_dict(), save_path)
        print('Saved model checkpoints into {}...'.format(save_path))
    
# splitting=np.array(splitting)
# accs = np.array(accs)

# t, c, k = interpolate.splrep(splitting, accs, s=0, k=4)
# xmin, xmax = splitting.min(), splitting.max()
# xx = np.linspace(xmin, xmax, 100)
# spline = interpolate.BSpline(t, c, k, extrapolate=False)

# fig, ax = plt.subplots()

# # ax.plot(splitting, accs, 'bo', label='Original points')
# ax.plot(xx, spline(xx))
# ax.grid()
# ax.legend(loc='best')


# ax.set_ylabel('Testing Accuracy')
# ax.set_xlabel('Testing Data Percentage')
# fig.savefig('11class-testing_acc.png', dpi=fig.dpi)
# f= open("11class-results.txt","w+")
# f.write("splitting percentage: ")
# f.write(str(splitting))
# f.write("accuracy is: ")
# f.write(str(accs))
# f.close() 
# #np.save('./model_complex.npy', manifold_net.covariance_block.conv[0].weight.detach().cpu().numpy(), )

