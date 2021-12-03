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



##### prepare test data ###################################################

FF=nc.Dataset('/glade/scratch/asheshc/theory-interp/QG/Transfer-Learning/set16_test/PSI_output.nc')
lat=np.asarray(FF['lat'])
lon=np.asarray(FF['lon'])

lead = 1

psi_test_input_Tr_torch, psi_test_label_Tr_torch, psi_test_label_Tr  = load_test_data(FF,lead)


def store_activations (Act,output_training,epoch,out,x1,x2,x3,x4,x5,x6,x7):

   Act[epoch,0,:,:,:,:] = x1.detach().cpu().numpy()
   Act[epoch,1,:,:,:,:] = x2.detach().cpu().numpy()
   Act[epoch,2,:,:,:,:] = x3.detach().cpu().numpy()
   Act[epoch,3,:,:,:,:] = x4.detach().cpu().numpy()
   Act[epoch,4,:,:,:,:] = x5.detach().cpu().numpy()
   Act[epoch,5,:,:,:,:] = x6.detach().cpu().numpy()
   Act[epoch,6,:,:,:,:] = x7.detach().cpu().numpy()

   output_training [epoch,:,:,:,:] = out.detach().cpu().numpy()

   return Act, output_training

def store_weights (net,epoch,hidden_weights_network,final_weights_network):

  hidden_weights_network[epoch,0,:,:,:,:] = net.hidden1.weight.data.cpu()
  hidden_weights_network[epoch,1,:,:,:,:] = net.hidden2.weight.data.cpu()
  hidden_weights_network[epoch,2,:,:,:,:] = net.hidden3.weight.data.cpu()
  hidden_weights_network[epoch,3,:,:,:,:] = net.hidden4.weight.data.cpu()
  hidden_weights_network[epoch,4,:,:,:,:] = net.hidden5.weight.data.cpu()
  hidden_weights_network[epoch,5,:,:,:,:] = net.hidden6.weight.data.cpu()
  final_weights_network[epoch,:,:,:,:] = net.hidden7.weight.data.cpu()

  return hidden_weights_network, final_weights_network


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


net = CNN()
net.load_state_dict(torch.load('./BNN.pt'))
net.cuda()
net.eval()

print('Model loaded')


### Freezing and Un-freezing the Layer #################

net.input_layer.weight.requires_grad = False
net.hidden1.weight.requires_grad = False
net.hidden2.weight.requires_grad = False
net.hidden3.weight.requires_grad = False
net.hidden4.weight.requires_grad = False
net.hidden5.weight.requires_grad = False
net.hidden6.weight.requires_grad = False
net.hidden7.weight.requires_grad = True


net.input_layer.bias.requires_grad = False
net.hidden1.bias.requires_grad = False
net.hidden2.bias.requires_grad = False
net.hidden3.bias.requires_grad = False
net.hidden4.bias.requires_grad = False
net.hidden5.bias.requires_grad = False
net.hidden6.bias.requires_grad = False
net.hidden7.bias.requires_grad = True


#### Red define placeholders for weights and activations ########
num_epochs = 15
num_samples = 2



Act = np.zeros([num_epochs,7,num_samples,64,192,96])
output_training = np.zeros([num_epochs,num_samples,2, 192, 96])
hidden_weights_network = np.zeros([num_epochs,6,64,64,5,5])
final_weights_network = np.zeros([num_epochs,2,64,5,5])




######### Begin Re-training #########################

loss_fn = nn.MSELoss()
optimizer = optim.SGD(filter(lambda p: p.requires_grad, net.parameters()), lr=0.001, momentum=0.9)
print('**** Number of Trainable Parameters in TLNN')
count_parameters(net)
batch_size = 100
num_epochs = 15
trainN = 7000
fileList_train=[]
mylist = [1]
for k in mylist:
  fileList_train.append ('/glade/scratch/asheshc/theory-interp/QG/Transfer-Learning/set'+str(k)+'_test'+'/PSI_output.nc')

print('****************')
print('Re-training starts')



for epoch in range(0, num_epochs):  # loop over the dataset multiple times

    running_loss = 0.0
    for loop in fileList_train:
     print('Training loop index',loop)

     psi_input_Tr_torch, psi_label_Tr_torch = load_train_data(loop, lead, trainN)

     for step in range(0,trainN,batch_size):
        # get the inputs; data is a list of [inputs, labels]
        indices = np.random.permutation(np.arange(start=step, stop=step+batch_size))
        input_batch, label_batch = psi_input_Tr_torch[indices,:,:,:], psi_label_Tr_torch[indices,:,:,:]
        print('shape of input', input_batch.shape)
        print('shape of output', label_batch.shape)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize
        output,_,_,_,_,_,_,_ = net(input_batch.cuda())
        loss = loss_fn(output, label_batch.cuda())
        loss.backward()
        optimizer.step()
        output_val,_,_,_,_,_,_,_ = net (psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,192,96]).cuda())
        val_loss = loss_fn(output_val, psi_test_label_Tr_torch[0:num_samples].reshape([num_samples,2,192,96]).cuda())

        if step % 100 == 0:    # print every 2000 mini-batches
            print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, step + 1, loss))
            print('[%d, %5d] val_loss: %.3f' %
                  (epoch + 1, step + 1, val_loss))
            running_loss = 0.0
    out,x1,x2,x3,x4,x5,x6,x7 = net (psi_test_input_Tr_torch[0:num_samples].reshape([num_samples,2,192,96]).cuda())

    hidden_weights_network, final_weights_network = store_weights(net,epoch,hidden_weights_network,final_weights_network)
    Act, output_training = store_activations (Act, output_training, epoch,out,x1,x2,x3,x4,x5,x6,x7)

torch.save(net.state_dict(), './TLNN_single_jet_beta_change_layer_7.pt')

print('TLNN Model Saved')

savenc_for_activations(Act,output_training,2,num_epochs,7,num_samples,64,192,96,'TLNN_singlejet_beta_change_Activations_layer_7_'+str(trainN)+'sample_size'+str(num_samples)+'_dt'+str(lead)+'.nc')

print('Saved Activations for TLNN')


matfiledata = {}
matfiledata[u'hidden_weights'] = hidden_weights_network
matfiledata[u'final_layer_weights'] = final_weights_network
hdf5storage.write(matfiledata, '.', 'TLNN_singlejet_beta_change_Weights_layer_7_'+str(trainN)+'sample_size'+str(num_samples)+'_dt'+str(lead)+'.mat', matlab_compatible=True)

print('Saved Weights for TLNN')



############# Auto-regressive prediction #####################
M=1000
autoreg_pred = np.zeros([M,2,192,96])

for k in range(0,M):

  if (k==0):

    out,_,_,_,_,_,_,_ = (net(psi_test_input_Tr_torch[k].reshape([1,2,192,96]).cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

  else:

    out,_,_,_,_,_,_,_ = (net(torch.from_numpy(autoreg_pred[k-1,:,:,:].reshape([1,2,192,96])).float().cuda()))
    autoreg_pred[k,:,:,:] = out.detach().cpu().numpy()

savenc(autoreg_pred, lon, lat, 'predicted_psi_TL_singlejet_layer_7_lead'+str(lead)+'.nc')
savenc(psi_test_label_Tr, lon, lat, 'truth_psi_TL_singlejet_layer_7_lead'+str(lead)+'.nc')
