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
require 'unsup'


params = {
    src_net = "../stored_kernels/autoencoders/ms_pacman_encoder_diag_0518.t7",
    src_data = "../stored_frames/frames_ms_pacman_0419.t7"
}

net = torch.load(params.src_net).model:double()
print (net)

data = torch.load(params.src_data).frames:double()


if arg[1] == nil then
    numImage = 100
else
    numImage = arg[1]
end
res = net:forward(data[numImage])
print (res:size())


original = image.toDisplayTensor{input=image.scale(data[numImage],res:size(2)*3, res:size(2)*3),
                                 padding=5,
                                 nrow = 2 }

reconstruction = image.toDisplayTensor{input=image.scale(res,res:size(2)*3, res:size(2)*3),
                                    padding=5,
                                    nrow = 2 }

image.display(data[numImage])
image.display(res)

image.display{image=original, legend="Original"}
image.display{image=reconstruction, legend="Reconstruction"}
