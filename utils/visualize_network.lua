package.path = package.path .. ";../dqn/?.lua"

require 'torch'
require 'nn'
require 'nngraph'
require 'image'
require 'cutorch'
require 'cunn'
require 'initenv'

params = {
    src_net = "../trained_networks_old/ms_pacman/DQN3_0_1_ms_pacman_FULL_Y_0430.t7",
    src_data = "../stored_frames/frames_ms_pacman_0419.t7",
    numConvolutions = 2
}

net = torch.load(params.src_net).model:double()
print (net)

data = torch.load(params.src_data).frames:double()


numImage = 500
input = data[numImage]
res = input

for i=1,params.numConvolutions*2+1 do
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
