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
    src_net = "../trained_networks/autoencoders/ms_pacman_0512.t7",
    src_data = "../stored_frames/frames_ms_pacman_0419.t7"
}

net = torch.load(params.src_net).model:double()
print (net)

data = torch.load(params.src_data).frames:double()


numImage = 100
res = net:forward(data[numImage])
print (res:size())

image.display(data[numImage])
image.display(res)
