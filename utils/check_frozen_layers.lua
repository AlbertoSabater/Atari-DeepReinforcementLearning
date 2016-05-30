package.path = package.path .. ";../dqn/?.lua"

require 'torch'
require 'nn'
require 'nngraph'
require 'image'
require 'cutorch'
require 'cunn'
require 'initenv'


params = {
    net1_src = "../trained_networks/DQN3_0_1_ms_pacman_FULL_Y_0527.model.t7",
    net2_src = "../stored_kernels/autoencoders/test_full_1Layer.t7"
}

net1 = torch.load(params.net1_src).model
net2 = torch.load(params.net2_src).network


print (net1)
print (net2)

layer1 = net1:get(2):get(3)
layer2 = net2:get(3)

print (layer1.weight[{1,1,3,1}])
print (layer2.weight[{1,1,3,1}])
