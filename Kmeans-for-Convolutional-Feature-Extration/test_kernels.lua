require 'torch'
require 'nn'
require 'nngraph'
require 'image'
require 'cutorch'





A = torch.load("../stored_kernels/kernels_ms_pacman_0418.t7",'binary')

print (A.kernels:size())
--print (A.counts:size())
--	print (A.counts:sum())

kw = A.kernels:resize(50,3,7,7)
--print (kw:size())
--image.display(img[1])


local net = nn.Sequential()
net:add(nn.SpatialConvolution(3, 50,
              7, 7,
              3, 3,1))

print (net:get(1))
print (net:get(1).weight:size())

--image.display(kw)
--image.display(net:get(1).weight)

net:get(1).weight = kw
print (net:get(1).weight:size())
--image.display(net:get(1).weight)



--[[
A = torch.load("../stored_frames/frames_ms_pacman_0418.t7",'binary')
fr = A.frames

print (fr:size())
--image.display(fr[1])

local data_dim = {4, 84, 84}
print (fr:size(1), fr:size(2), fr:size(3), fr:size(4))
]]


--[[
t = {}
table.insert(t,kw[1])
table.insert(t,kw[2])

merge=nn.Sequential()
        :add(nn.JoinTable(1))
        :add(nn.View(-1, 3, 7, 7))


output = merge:forward(t)
print (kw[1]:size())
print(output:size())

--image.display(kw[2])
--image.display(output[2])
]]
