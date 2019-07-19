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

# Parameters for data loading
params_train = {'batch_size': 1,
          'shuffle': True,
          'num_workers': 1}

max_epochs = 100

sample_images = ['c'+str(i)+"_0.npy" for i in range(1, 12)]
label_list = [{'c'+str(i)+"_0.npy": i-1} for i in range(1, 12)]

labels = {}
for i in label_list:
    labels.update(i)

data_set = model.Dataset(sample_images, labels)

#Default device
device = torch.device('cuda: 0' if torch.cuda.is_available() else 'cpu')
logger = setup_logger("Comparison")

def test(model, device, test_loader):
    
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
            #For confusion matrix
            correct += pred.eq(target.view_as(pred)).sum().item()
    
    return 100. * correct / len(test_loader.dataset)


manifold_net = model.TestNet().cuda()
init_params = manifold_net.parameters()
model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
params = sum([np.prod(p.size()) for p in model_parameters])
logger.info("Model Parameters: "+str(params))
save_path = os.path.join('./save/', 'peter-batch[150]-lr[0.008]-acc[97.893]-11class-model-15-17.ckpt')
manifold_net.load_state_dict(torch.load(save_path))
np.random.seed(42222222)
optimizer = optim.Adam(manifold_net.parameters(), lr=0.05)
criterion = nn.CrossEntropyLoss()

def classification(out,desired):
    _, predicted = torch.max(out, 1)
    total = desired.shape[0]
    correct = (predicted == desired).sum().item()
    return correct


from matplotlib.colors import hsv_to_rgb


train_generator = torch.utils.data.DataLoader(dataset=data_set, **params_train)
for it,(local_batch, local_labels) in enumerate(train_generator):
    batch = torch.tensor(local_batch, requires_grad=True).cuda()
    print(batch.shape)
    
    
    ####
    batch[:, 1,...] = torch.ones(batch[:, 1, ...].shape)
    ####
    
    
    
    out, x1, x2, x3, x4, x5, x6 = manifold_net(batch)
    label = local_labels.item()
    
    fig = plt.figure(figsize=(6, 6))
    img = batch[0, :, 0, :, :].cpu().detach().numpy()
    img = img.transpose((1, 2, 0))
    img = np.insert(img, 1, 0.5, axis=2)
    img = hsv_to_rgb(img)
    plt.imshow(img)
    fig.savefig('./visualize/class'+str(label)+'_original.png', dpi=fig.dpi)
    
    fig = plt.figure(figsize=(6, 6))
    
    
    sub1 = fig.add_subplot(231) 
    img1 = x1[0, :, 0, :, :].cpu().detach().numpy()
    imshape = img1.shape
    img1 = img1.transpose((1, 2, 0))
    img1 = np.insert(img1, 1, 0.5, axis=2)
#     img1[:, :, 0] += np.pi
    img1 = hsv_to_rgb(img1)
    plt.imshow(img1)

    sub2 = fig.add_subplot(232)
    img2 = x2[0, :, -1, :, :].cpu().detach().numpy()
    imshape = img2.shape
    img2 = img2.transpose((1, 2, 0))
    img2 = np.insert(img2, 1, 0.5, axis=2)
#     img2[:, :, 0] += np.pi
    img2 = hsv_to_rgb(img2)
    plt.imshow(img2)
    
    sub3 = fig.add_subplot(233)
    img3 = x3[0, 10, :, :].cpu().detach().numpy()
    imshape = img3.shape
    plt.imshow(img3, cmap='gray')
    
    sub4 = fig.add_subplot(234)
    img4 = x4[0, 10, :, :].cpu().detach().numpy()
    imshape = img4.shape
    plt.imshow(img4, cmap='gray')
    
    sub5 = fig.add_subplot(235)
    img5 = x5[0, 10, :, :].cpu().detach().numpy()
    imshape = img5.shape
    plt.imshow(img5, cmap='gray')
    
    sub6 = fig.add_subplot(236)
    img6 = x6[0, :, :, :].cpu().detach().numpy().reshape(x6.shape[1], 1)
    print(img6.shape)
    imshape = img6.shape
    plt.imshow(img6, 'gray')
    
    fig.savefig('./visualize/class'+str(label)+'.png', dpi=fig.dpi)
    
#     sub2 = fig.add_subplot(232)
#     sub2.set_title('fp, the derivation of f')
#     sub2.plot(t, fp(t))
#     t = np.arange(-3.0, 2.0, 0.02)
#     sub3 = fig.add_subplot(233)
#     sub3.set_title('The function g')
#     sub3.plot(t, g(t))
#     t = np.arange(-0.2, 0.2, 0.001)
#     sub4 = fig.add_subplot(234)
#     sub4.set_title('A closer look at g')
#     sub4.set_xticks([-0.2, -0.1, 0, 0.1, 0.2])
#     sub4.set_yticks([-0.15, -0.1, 0, 0.1, 0.15])
#     sub4.plot(t, g(t))
#     sub5 = fig.add_subplot(235)
#     sub6 = fig.add_subplot(236)
#     fig.savefig('1edede.png', dpi=fig.dpi)
    
    
#     sub1 = fig.add_subplot(221) # instead of plt.subplot(2, 2, 1)
#     sub1.set_title('complex_conv1') # non OOP: plt.title('The function f')
#     sub1.plot(t, f(t))
#     sub2 = fig.add_subplot(222, axisbg="lightgrey")
#     sub2.set_title('manifold_ReLU')
#     sub2.plot(t, fp(t))
#     sub3 = fig.add_subplot(223)
#     sub3.set_title('complex_conv2')
#     sub3.plot(t, g(t))
#     sub4 = fig.add_subplot(224, axisbg="lightgrey")
#     sub4.set_title('manifold_ReLU')
#     sub4.plot(t, g(t))
#     fig.savefig('10class-testing_acc1.png', dpi=fig.dpi)


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
# fig.savefig('10class-testing_acc1.png', dpi=fig.dpi)
# f= open("10class-results1.txt","w+")
# f.write("splitting percentage: ")
# f.write(str(splitting))
# f.write("accuracy is: ")
# f.write(str(accs))
# f.close() 


# #11 class
# splitting = [0.5, 0.6, 0.7, 0.8, 0.9]
# accs = []


# for i in splitting:
#     logger.info("split training/testing: "+str(1-i)+ "/"+ str(i))
#     manifold_net = model.ManifoldNetComplex_11().cuda()


#     init_params = manifold_net.parameters()
#     model_parameters = filter(lambda p: p.requires_grad, manifold_net.parameters())
#     params = sum([np.prod(p.size()) for p in model_parameters])
#     logger.info("Model Parameters: "+str(params))
#     #manifold_net.load_state_dict(torch.load('models/pretrained.pt'))
#     optimizer = optim.Adam(manifold_net.parameters(), lr=0.05)
#     criterion = nn.CrossEntropyLoss()
#     scheduler = ReduceLROnPlateau(optimizer, 'min', patience=15)

#     def classification(out,desired):
#         _, predicted = torch.max(out, 1)
#         total = desired.shape[0]
#         correct = (predicted == desired).sum().item()
#         return correct

#     # Training...
#     print('Starting training...')
#     validation_accuracy = []
#     highest=0
#     #split = 0
#     try:
#         #for train_idx, test_idx in data_utils.k_folds(n_splits=10, n_samples=(14557)):
#             np.random.seed(42222222)
#             idx = np.random.permutation(14557+1159)
#             train_end = math.ceil((14557+1159)*(1-i))
#             train_idx = idx[:train_end]
#             test_idx = idx[train_end:]   
#             data_train = data.Subset(data_set_11,indices=train_idx)
#             data_test = data.Subset(data_set_11,indices=test_idx)

#             train_generator = torch.utils.data.DataLoader(dataset=data_train, **params_train)
#             test_generator = torch.utils.data.DataLoader(dataset=data_test, **params_val)
#             epoch_validation_history = []
#             for epoch in range(max_epochs):
#                 print('Starting Epoch ', epoch, '...')
#                 loss_sum = 0
#                 start = time.time()
#                 train_acc = 0
#                 for it,(local_batch, local_labels) in enumerate(train_generator):
#                     batch = torch.tensor(local_batch, requires_grad=True).cuda()
                    
#                     batch[:, 0, :, :] = torch.ones(batch[:, 0, :, :].shape)
                    
                    
#                     optimizer.zero_grad()
#                     out = manifold_net(batch)
#                     train_acc += classification(out, local_labels.cuda()) 

#                     loss = criterion(out, local_labels.cuda())
#                     loss.backward()
#                     optimizer.step()
#                 logger.info("Epoch: "+str(epoch)+"Training accuracy: "+str(train_acc / len(train_generator.dataset)*100.))

#                 acc = test(manifold_net, device, test_generator)
#                 scheduler.step(acc)
#                 if acc > highest:
#                     highest=acc
#                     save_path = os.path.join('./save/', 'split[{split}]-11class-model-1s.ckpt'.format(split=i))
#                     torch.save(manifold_net.state_dict(), save_path)
#                     print('Saved model checkpoints into {}...'.format(save_path))
#                 logger.info("Epoch: "+str(epoch)+"Testing accuracy is "+str(acc))
#                 end = time.time()
#                 print('Epoch Time:', end-start)
#             accs.append(highest)
#             manifold_net = model.ManifoldNetComplex_11().cuda()
#             optimizer = optim.Adam(manifold_net.parameters(), lr=0.05)
#             criterion = nn.CrossEntropyLoss()
#     except KeyboardInterrupt:
#         pass


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
# fig.savefig('11class-testing_acc1s.png', dpi=fig.dpi)
# f= open("11class-results1s.txt","w+")
# f.write("splitting percentage: ")
# f.write(str(splitting))
# f.write("accuracy is: ")
# f.write(str(accs))
# f.close() 