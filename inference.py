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
from data_loader import load_test_data
from data_loader import load_train_data


##### prepare test data ###################################################

FF=nc.Dataset('/glade/scratch/asheshc/theory-interp/QG/Transfer-Learning/set16_test/PSI_output.nc')
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

        return out

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
net = CNN()
net.load_state_dict(torch.load('./BNN.pt'))
net.cuda()
net.eval()

print('Model loaded')
M=1000
autoreg_pred = np.zeros([M,2,192,96])

for k in range(0,M):

  if (k==0):

    autoreg_pred[k,:,:,:] = (net(psi_test_input_Tr_torch[k].reshape([1,2,192,96]).cuda())).detach().cpu().numpy()

  else:

    autoreg_pred[k,:,:,:] = (net(torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,2,192,96])).float().cuda())).detach().cpu().numpy()

savenc(autoreg_pred, lon, lat, 'predicted_psi_singlejet_beta_change_BNN_lead'+str(lead)+'.nc')
savenc(psi_test_label_Tr, lon, lat, 'truth_psi_singlejet_beta_change_BNN_lead'+str(lead)+'.nc')






