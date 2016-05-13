package.path = package.path .. ";../dqn/?.lua"

require 'torch'
require 'nn'
require 'nngraph'
require 'nnutils'
require 'image'
require 'Scale'
require 'nnutils'
require 'cutorch'
require 'initenv'
require 'image'

params = {
    src_net = "../trained_networks_old/ms_pacman/DQN3_0_1_ms_pacman_FULL_Y_0430.t7",
    src_data = "../stored_frames/frames_ms_pacman_0419.t7"
}

net = torch.load(params.src_net).model:double()
print (net)

data = torch.load(params.src_data).frames:double()


numImage = 100
res = net:get(2):forward(data[numImage])
res = net:get(3):forward(res)
print (res:size())

image.display(res)
