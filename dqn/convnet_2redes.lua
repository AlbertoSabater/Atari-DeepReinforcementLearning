--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

require "initenv"

function create_network(args)

    local net = nn.Sequential()
    local aux = nn.Sequential()
    local trained_network
    local convLayer = nn.SpatialConvolution

if args.load_weights == 1 then              -- Load convolutional network with trained features
    print ("Loading convolutional network with trained features")
    local K = torch.load(args.weights_src,'binary')
    net = K.network

    net:insert(nn.Reshape(unpack(args.input_dims)), 1)
else


--=============================================================================

  aux = nn.Sequential()
  aux:add(nn.Reshape(unpack(args.input_dims)))
  aux:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                      args.filter_size[1], args.filter_size[1],
                      args.filter_stride[1], args.filter_stride[1],1))
  aux:add(args.nl())

  -- Add convolutional layers
  for i=1,(#args.n_units-1) do
      -- second convolutional layer
      aux:add(convLayer(args.n_units[i], args.n_units[i+1],
                          args.filter_size[i+1], args.filter_size[i+1],
                          args.filter_stride[i+1], args.filter_stride[i+1]))
      aux:add(args.nl())
  end


--=============================================================================

  net:add(nn.Reshape(unpack(args.input_dims)))

  -- LOADING TRAINED NETWORK TO COMBINE WITH A NEW ONE
  if args.trained_kernels_net ~= nil and args.trained_kernels_net ~= "" then
    local K = torch.load(args.trained_kernels_net,'binary')
    trained_network = K.network
    trained_network:insert(nn.Reshape(unpack(args.input_dims)), 1)

    -- CREATE MAIN NETWORK WITHOUT THE TRAINED_NETWORK PARAMETERS
    net:add(convLayer(args.hist_len*args.ncols, args.n_units[1] - trained_network:get(2).nOutputPlane,
                        args.filter_size[1], args.filter_size[1],
                        args.filter_stride[1], args.filter_stride[1],1))
    net:add(args.nl())

    for i=1,(#args.n_units-1) do

        if (i+1)*2 < trained_network:size() then    -- THIS LAYER EXIST IN trained_network
            -- second convolutional layer
            net:add(convLayer(net:get(i*2).nOutputPlane, args.n_units[i+1]  - trained_network:get((i+1)*2).nOutputPlane,
                                args.filter_size[i+1], args.filter_size[i+1],
                                args.filter_stride[i+1], args.filter_stride[i+1]))
            net:add(args.nl())
        else
            -- second convolutional layer
            net:add(convLayer(args.n_units[i], args.n_units[i+1],
                                args.filter_size[i+1], args.filter_size[i+1],
                                args.filter_stride[i+1], args.filter_stride[i+1]))
            net:add(args.nl())
        end

    end

  else
    --[[
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
    ]]
    net = aux
  end

end

    local nel
    if args.gpu >= 0 then
        -- net:add(nn.Transpose({4,3},{3,2},{2,1})) -> deprecated with CUDA
        if aux == nil then
          nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims)):cuda()):nElement()
        else
          nel = aux:cuda():forward(torch.zeros(1,unpack(args.input_dims)):cuda()):nElement()
        end
    else
        if aux == nil then
          nel = bet:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
        else
          nel = aux:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
        end
    end

if args.only_conv ~= 1 then

    -- reshape all feature planes into a vector per example
    net:add(nn.Reshape(nel))

    -- fully connected layer
print (args.n_hid[1])
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
        if trained_network ~= nil then
          trained_network:cuda()
        end
    end
    if args.verbose >= 2 then
        print("Main network->", net)
        print ("Trained network->", trained_network)
        print('Convolutional layers flattened output size:', nel)
    end



    return net, trained_network
end
