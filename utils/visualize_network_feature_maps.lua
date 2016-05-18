package.path = package.path .. ";../dqn/?.lua"

require 'torch'
require 'nn'
require 'nngraph'
require 'image'
require 'cutorch'
require 'cunn'
require 'initenv'

params = {
    src_net = "../stored_kernels/autoencoders/ms_pacman_encoder_2_none_0518.t7",
    src_data = "../stored_frames/frames_ms_pacman_0419.t7",
    numConvolutions = 2,
    firstLayer = 1,
    numImage  = 100
}

cmd = torch.CmdLine()
cmd:option('-src_net', params.src_net, 'path to network')
cmd:option('-src_data', params.src_data, 'path to data')
cmd:option('-numConvolutions', params.numConvolutions, 'Number of convolutions')
cmd:option('-firstLayer', params.firstLayer, 'Number of first layer')
cmd:option('-numImage', params.numImage, 'Image to visualize')
options = cmd:parse(arg)

for k, v in pairs(options) do
    print(k, v)
end


net = torch.load(options.src_net).network:double()
print (net)

data = torch.load(options.src_data).frames:double()


input = data[options.numImage]
res = input

for i=options.firstLayer,options.numConvolutions*2+(options.firstLayer-1) do
    res = net:get(i):forward(res)
    print (net:get(i))
end


original = image.toDisplayTensor{input=image.scale(input,input:size(2)*3, input:size(2)*3),
                                 padding=5,
                                 nrow = 2 }

reconstruction = image.toDisplayTensor{input=image.scale(res,input:size(2)*2, input:size(2)*2),
                                    padding=5,
                                    nrow = 4 }

image.display({image=original, leyend="Original"})
image.display({image=reconstruction, leyend="Reconstruction"})
