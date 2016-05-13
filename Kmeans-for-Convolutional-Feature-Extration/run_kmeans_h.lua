#!/usr/bin/env th
--
-- An Analysis of Single-Layer Networks in Unsupervised Feature Learning
-- by Adam Coates et al. 2011
--
-- The original MatLab code can be found in http://www.cs.stanford.edu/~acoates/
-- Tranlated to Lua/Torch7
--

package.path = '/home/asabater/Atari-DeepReinforcementLearning/dqn/?.lua;' .. package.path

require("xlua")
require("image")
require("unsup")
require("kmeans")
require("extract")
require("train-svm")
require("initenv")

torch.manualSeed(1)
torch.setdefaulttensortype('torch.FloatTensor')
opt = {
   whiten = false,
}
local date = os.date("%m%d")


local game = "ms_pacman"
local network = "convnet_paper1_bigger"

local msg, err = pcall(require, network)
network= err

args = {}
args.hist_len = 4
args.ncols = 1
args = network(args)


local data = torch.load("../stored_frames/frames_" .. game .. "_0419.t7",'binary')


local fr = data.frames
-- set parameters
local data_dim = {fr:size(2), fr:size(3), fr:size(4)}
local trsize = fr:size(1)
local numPatches = 50000
local featureRelation = 2

local numLayers = 5


local net = nn.Sequential()
local convLayer = nn.SpatialConvolution

for i=1,math.min(numLayers,#args.n_units) do


    -- get net frames
    print("==> Get frames")


    nextData = {}
    for i=1,fr:size(1) do
        table.insert(nextData, net:forward(fr[i]))
    end

    print ("==> Merge")
    --print (net:forward(fr[1]):size())
    --print (net:forward(fr[1]):size(1),net:forward(fr[1]):size(2),net:forward(fr[1]):size(3))
    local merge=nn.Sequential()
                  :add(nn.JoinTable(1))
                  :add(nn.View(-1,net:forward(fr[1]):size(1),net:forward(fr[1]):size(2),net:forward(fr[1]):size(3)))
    local currentData = merge:forward(nextData)
    --print(currentData:size())


    print("==> extract patches")
    local kSize = args.filter_size[i]
    local data_dim = { net:forward(fr[1]):size(1),net:forward(fr[1]):size(2),net:forward(fr[1]):size(3) }
    local patches = torch.zeros(numPatches, kSize*kSize*data_dim[1])

    for i = 1,numPatches do
       xlua.progress(i,numPatches)
       local r = torch.random(data_dim[2] - kSize + 1)
       local c = torch.random(data_dim[3] - kSize + 1)
       patches[i] = currentData[{math.fmod(i-1,trsize)+1,{},{r,r+kSize-1},{c,c+kSize-1}}]
       if opt.whiten then
         patches[i] = patches[i]:add(-patches[i]:mean())
         patches[i] = patches[i]:div(math.sqrt(patches[i]:var()+10))
       end
    end


    if opt.whiten then
       print("==> whiten patches")
       local function zca_whiten(x)
          local dims = x:size()
          local nsamples = dims[1]
          local ndims    = dims[2]
          local M = torch.mean(x, 1)
          local D, V = unsup.pcacov(x)
          x:add(torch.ger(torch.ones(nsamples), M:squeeze()):mul(-1))
          local diag = torch.diag(D:add(0.1):sqrt():pow(-1))
          local P = V * diag * V:t()
          x = x * P
          return x, M, P
       end
       patches, M, P = zca_whiten(patches)
    end


    local currentCentroids = args.n_units[i]/featureRelation
    local ncentroids = currentCentroids
    local centroidsFound = 0
    local kenels, kernels_v2
    local counts, counts_v2

    while centroidsFound == 0 do    -- Iterate until get all centroids
      print("==> find clusters", ncentroids)
      kernels, counts = unsup.kmeans_modified(patches, ncentroids, nil, 0.1, 1, 1000, nil, true)
      torch.save("../stored_kernels/kernels_" .. game .. "_" .. date .. "v0.t7", { kernels = kernels, counts = counts, patches = patches })
      print(#counts)
      --print (kernels:size())



      print("==> select distinct features")
      local j = 0
      for i = 1,ncentroids do
         if counts[i] > 0 then
            j = j + 1
            kernels[{j,{}}] = kernels[{i,{}}]
            counts[j] = counts[i]
         end
      end
      kernels_v2 = kernels[{{1,j},{}}]
      counts_v2  = counts[{{1,j}}]
      print(#counts_v2)
      --print (kernels_v2:size())

      --print (kernels:size())
      --image.display(kernels[14]:resize(4,7,7))
      torch.save("../stored_kernels/kernels_" .. game .. "_" .. date .. ".t7", { kernels = kernels_v2, counts = counts_v2 })

len = #counts_v2
      if len[1] == currentCentroids then   -- kernels found
        centroidsFound = 1
      elseif len[1] < currentCentroids then   -- more kernels
centroidsFound = 1
        ncentroids = ncentroids + 1
      else
        centroidsFound = 1
      end
    end

    -- add layer
    print("==> Add layer")
    if i == 1 then  -- First convolutional layer
        net:add(convLayer(args.hist_len*args.ncols, len[1],
                        args.filter_size[i], args.filter_size[i],
                        args.filter_stride[i], args.filter_stride[i],1))
    else
        net:add(convLayer(net:get((i-1)*2 -1).nOutputPlane, len[1],
                        args.filter_size[i], args.filter_size[i],
                        args.filter_stride[i], args.filter_stride[i]))
    end
    net:add(args.nl())

    -- set weights

    print (net)
    --net:get(i*2-1).weight = kernels:resize(net:get(i*2-1).weight:size())
    net:get(i*2-1).weight = kernels_v2

    torch.save("../stored_kernels/kernels_net_" .. game .. "_" .. date .. ".t7", { network = net })

end


local win = nil
for i=1,32 do

    --print(net:forward(fr[100]):size())
    --win = image.display({image=net:forward(fr[100])[i]})

end

    --win = image.display({image=net:get(1):forward(fr[100])})
