--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"

function create_network(args)

    local net = nn.Sequential()
    local convLayer = nn.SpatialConvolution

if args.load_weights == 1 then              -- Load convolutional network with trained features
    print ("Loading convolutional network with trained features")
    local K = torch.load(args.weights_src,'binary')
    net = K.network
    
    net:insert(nn.Reshape(unpack(args.input_dims)), 1)
else 
   net:add(nn.Reshape(unpack(args.input_dims)))

    --- first convolutional layer


    --[[ deprecated with CUDA
    if args.gpu >= 0 then
        net:add(nn.Transpose({1,2},{2,3},{3,4}))
        convLayer = nn.SpatialConvolutionCUDA
    end
    ]]
    
    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())

    -- Add convolutional layers
    for i=1,(#args.n_units-1) do
        -- second convolutional layer
        net:add(convLayer(args.n_units[i], args.n_units[i+1],
                            args.filter_size[i+1], args.filter_size[i+1],
                            args.filter_stride[i+1], args.filter_stride[i+1]))
        net:add(args.nl())
    end
end
  
   

    local nel
    if args.gpu >= 0 then
        -- net:add(nn.Transpose({4,3},{3,2},{2,1})) -> deprecated with CUDA
        nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                :cuda()):nElement()
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

if args.only_conv ~= 1 then
	
    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
    net:add(nn.Linear(nel, args.n_hid[1]))
    net:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        net:add(nn.Linear(args.n_hid[i], last_layer_size))
        net:add(args.nl())
    end

    -- add the last fully connected layer (to actions)
    net:add(nn.Linear(last_layer_size, args.n_actions))

end


    if args.gpu >=0 then
        net:cuda()
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end

    
    --[[
    print ("==========================================================================================")
    
    local msg, err = pcall(require, "/tmp/trained_networks/DQN3_0_1_breakout_FULL_Y_0405.params.t7")
    if not msg then
        print("Loading trained network", "/tmp/trained_networks/DQN3_0_1_breakout_FULL_Y_0405.params.t7")
        -- try to load saved agent
        local err_msg, exp = pcall(torch.load, "/tmp/trained_networks/DQN3_0_1_breakout_FULL_Y_0405.params.t7")
        if not err_msg then
            error("Could not find network file ")
        end
    end
    
    print ("==========================================================================================")
    for i,v in ipairs(exp) do 
      print("AAAAA")
      print (v) 
    end
    
    
    print (net.weight)
    net.w = exp.w
    print ("==========================================================================================")
    print (net.w)
    ]]
    
    return net
end
