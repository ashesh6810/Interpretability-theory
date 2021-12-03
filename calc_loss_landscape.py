import numpy as np
import torch
print(torch.__version__)

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchinfo import summary
import sys
import netCDF4 as nc
from saveNCfile import savenc
from saveNCfile_for_activations import savenc_for_activations
from data_loader import load_test_data
from data_loader import load_train_data
from prettytable import PrettyTable
from count_trainable_params import count_parameters
import hdf5storage
from scipy.io import savemat

FF=nc.Dataset('/glade/scratch/asheshc/theory-interp/QG/Transfer-Learning/case1/PSI_output.nc')
lat=np.asarray(FF['lat'])
lon=np.asarray(FF['lon'])

lead = 1

psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr  = load_test_data(FF,lead)



class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_layer = (nn.Conv2d(2, 64, kernel_size=5, stride=1, padding='same'))
#        torch.nn.init.normal_(self.input.weight, mean=0, std =0.01)
        self.hidden1 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))

        self.hidden2 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden3 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden4 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden5 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden6 = (nn.Conv2d(64, 64, kernel_size=5, stride=1, padding='same' ))
        self.hidden7 = (nn.Conv2d(64, 2, kernel_size=5, stride=1, padding='same' ))



#        torch.nn.init.normal_(self.hidden1.weight, mean=0, std =0.01)
    def forward(self,x):

        x1 = F.relu (self.input_layer(x))
        x2 = F.relu (self.hidden1(x1))
        x3 = F.relu (self.hidden2(x2))
        x4 = F.relu (self.hidden3(x3))
        x5 = F.relu (self.hidden4(x4))
        x6 = F.relu (self.hidden5(x5))
        x7 = F.relu (self.hidden6(x6))
        out = (self.hidden7(x7))

        return out, x1, x2, x3, x4, x5, x6, x7




device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

net = CNN()
net.load_state_dict(torch.load('./TLNN.pt'))
net.cuda()
net.eval()
loss_fn = nn.MSELoss()
num_samples = 20
loss_land = np.zeros([41,41])


#### Identify the layer for perturbation ######

perturb_layer = net.hidden7.weight.data.cpu().numpy()
print('Shape of layer',np.shape(perturb_layer))
print('perturbation layer taken to CPU')
###############################################

###### Pertrubation in two random direction ########
direction_w = np.random.randn(2,64,5,5)
direction_2_w = np.random.randn(2,64,5,5)
########################################


###### Gramschmdt orthonormalization #######

for i in range(2):
    for j in range(64):
        direction_w[i,j,:,:] = (direction_w[i,j,:,:]/np.linalg.norm(direction_w[i,j,:,:]))*np.linalg.norm(perturb_layer[i,j,:,:])
        direction_2_w[i,j,:,:] = (direction_2_w[i,j,:,:]/np.linalg.norm(direction_2_w[i,j,:,:]))*np.linalg.norm(perturb_layer[i,j,:,:])

print('Gram-Schmidt Normalization worked')

##############################


#### Define energy landscape and iterate over perturbation ######

loss_land = np.zeros([41,41])


for i in range(41):
  
   for j in range (41):
       
       print('outer iteration',i)
       alpha = 0.1*(i-20)
       alpha2 = 0.1*(j-20)
    
   
       with torch.no_grad():
        net.hidden7.weight = torch.nn.Parameter((torch.from_numpy(perturb_layer + alpha*direction_w + alpha2*direction_2_w)).float().cuda()) 
   

       pred, _,_,_,_,_,_,_ = net (psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,192,96]).cuda()) 
       loss = loss_fn(pred, psi_test_label_Tr_torch[0:num_samples].reshape([num_samples,2,192,96]).cuda())    
       loss_land[i,j] = loss


savemat('TLNN_Landscape_layer7'+'_dt'+str(lead)+'.mat',dict([('Loss',loss_land)]))

print('energy landscape saved')


 
