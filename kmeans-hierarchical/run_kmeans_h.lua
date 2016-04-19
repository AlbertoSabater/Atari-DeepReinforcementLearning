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
local network = "convnet_paper1"

local msg, err = pcall(require, network) 
network= err

args = str_to_table('lr=0.00025,ep=1,ep_end=0.1,ep_endt=replay_memory,discount=0.99,hist_len=4,learn_start=50000,replay_memory=1000000,update_freq=4,n_replay=1,network="convnet_paper1",preproc="net_downsample_2x_full_y",state_dim=7056,minibatch_size=32,rescale_r=1,ncols=1,bufferSize=512,valid_size=500,target_q=10000,clip_delta=1,min_reward=-1,max_reward=1,save_frames=0,only_conv=1')
args.input_dims = {args.hist_len*args.ncols, 84, 84}
args.gpu = -1
args.verbose = 0

network, args = network(args)

    
    
    
    
    
    
    
local data = torch.load("../stored_frames/frames_" .. game .. "_0419.t7",'binary')


local fr = data.frames
-- set parameters
local data_dim = {fr:size(2), fr:size(3), fr:size(4)}
local trsize = fr:size(1)
local numPatches = 50000


local net = nn.Sequential()
local convLayer = nn.SpatialConvolution

for i=1,#args.n_units do
      
      
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
    
    --print ("ksize", kSize)
    --print ("data_dim", unpack(data_dim))
    --print ("patches", patches:size())
    for i = 1,numPatches do
       xlua.progress(i,numPatches)
       local r = torch.random(data_dim[2] - kSize + 1)
       local c = torch.random(data_dim[3] - kSize + 1)
       patches[i] = currentData[{math.fmod(i-1,trsize)+1,{},{r,r+kSize-1},{c,c+kSize-1}}]
       patches[i] = patches[i]:add(-patches[i]:mean())
       patches[i] = patches[i]:div(math.sqrt(patches[i]:var()+10))
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


    local ncentroids = args.n_units[i]
    print("==> find clusters")
    kernels, counts = unsup.kmeans_modified(patches, ncentroids, nil, 0.1, 1, 1000, nil, true)
    torch.save("../stored_kernels/kernels_" .. game .. "_" .. date .. "v0.t7", { kernels = kernels, counts = counts, patches = patches })
    --print(counts)
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
    -- just select 1600 kernels for now


    --print (kernels:size())
    --image.display(kernels[14]:resize(4,7,7))
    torch.save("../stored_kernels/kernels_" .. game .. "_" .. date .. ".t7", { kernels = kernels_v2, counts = counts_v2 })


    -- add layer
    print("==> Add layer")
    if i == 1 then
        net:add(convLayer(args.hist_len*args.ncols, args.n_units[i],
                        args.filter_size[i], args.filter_size[i],
                        args.filter_stride[i], args.filter_stride[i],1))
    else 
        net:add(convLayer(args.n_units[i-1], args.n_units[i],
                        args.filter_size[i], args.filter_size[i],
                        args.filter_stride[i], args.filter_stride[i],1))
    end
    net:add(args.nl())
    
    -- set weights
  
    print (net)
    net:get(i*2-1).weight = kernels:resize(net:get(i*2-1).weight:size())
    
    torch.save("../stored_kernels/kernels_net_" .. game .. "_" .. date .. ".t7", { network = net })

end


local win = nil
for i=1,32 do
  
    --print(net:forward(fr[100]):size())
    --win = image.display({image=net:forward(fr[100])[i]})
  
end

    win = image.display({image=net:get(1):forward(fr[100])})

