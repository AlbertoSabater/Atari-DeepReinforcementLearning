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
    layer = 1,
    numImage  = 100,
    scaleFilter = 1,
    scaleRes = 1,
    numDimension = 1
}

cmd = torch.CmdLine()
cmd:option('-src_net', params.src_net, 'path to network')
cmd:option('-layer', params.layer, 'layer to visualize his filters')
cmd:option('-numImage', params.numImage, 'Image to visualize')
cmd:option('-scaleFilter', params.scaleFilter, 'Filter scale factor')
cmd:option('-scaleRes', params.scaleRes, 'Result scale factor')
cmd:option('-numDimension', params.numDimension, 'Dimension to display')
options = cmd:parse(arg)

for k, v in pairs(options) do
    print(k, v)
end


net = torch.load(options.src_net).network:double()
print (net)


filters = net:get(options.layer).weight

--print (filters[{{},1,{},{}}])


filters = image.toDisplayTensor{input=image.scale(filters[{{},options.numDimension,{},{}}], filters:size(3)*options.scaleFilter),
                                    padding=5,
                                    nrow = 4 }

--image.toDisplayTensor()
print (filters:size())
--image.scale()
image.display({image=image.scale(filters, filters:size(1)*options.scaleRes), leyend="Filters", mode="bicubic"})
