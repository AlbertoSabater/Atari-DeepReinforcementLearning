--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

--require "initenv"

function create_network(args)

    local net = nn.Sequential()
    local convLayer = nn.SpatialConvolution

    local convLayers = nn.Sequential()

    if args.load_weights == 1 then              -- Load convolutional network with trained features
        print ("Loading convolutional network with trained features")
        local K = torch.load(args.weights_src,'binary')
        convLayers = K.network

        net:insert(nn.Reshape(unpack(args.input_dims)), 1)
        net:add(convLayers)
    else
        net:add(nn.Reshape(unpack(args.input_dims)))

        if args.load_net_kernels == 1 and args.trained_kernels_net ~= nil and args.trained_kernels_net ~= "" then

      --==============================================================================
      -- LOAD SPECIFIED NUMBER OF LAYERS TRAINED NETWORK
            local K = torch.load(args.trained_kernels_net,'binary')
            local aux = K.network
            local net1 = nn.Sequential()

print ("AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA", aux:size())
            if args.num_layers == -1 or args.num_layers*2 >=   aux:size() then
                net1 = aux:clone()
            else
                for i=1,args.num_layers*2 do   -- Load the specified number or layers with their ReLU's
                    net1:add(aux:get(i))
                end
            end

            -- Freezing net1
            if args.freeze_kernels == 1 then
print ("Freezing kernels")
                net1.accGradParameters = function() end
                net1.updateParameters = function() end
            end

      --==============================================================================
      -- CREATE MAIN NETWORK WITHOUT THE net1 NUMBER FEATURES
            net2 = nn.Sequential()

            if args.n_units[1] - net1:get(1).nOutputPlane > 0 then
                net2:add(convLayer(args.hist_len*args.ncols, args.n_units[1] - net1:get(1).nOutputPlane,
                                    args.filter_size[1], args.filter_size[1],
                                    args.filter_stride[1], args.filter_stride[1],1))
                net2:add(args.nl())
            end

            local index
            for i=1,(#args.n_units-1) do
                if (i+1)*2 <= net1:size() then    -- THIS LAYER EXIST IN trained_network
                    -- second convolutional layer
                    if args.n_units[i+1] - net1:get((i+1)*2-1).nOutputPlane > 0 then
                        net2:add(convLayer(net2:get(i*2-1).nOutputPlane, args.n_units[i+1] - net1:get((i+1)*2-1).nOutputPlane,
                                            args.filter_size[i+1], args.filter_size[i+1],
                                            args.filter_stride[i+1], args.filter_stride[i+1]))
                        net2:add(args.nl())
                    end
                else
                  index = i
                end
            end
      --==============================================================================


            if net2:size() > 0 then
                parallel_model = nn.Concat(2)  -- model that concatenates net1 and net2
                parallel_model:add(net1)
                parallel_model:add(net2)
                convLayers:add(parallel_model)
            else
                convLayers:add(net1)
            end

      --==============================================================================
      -- ADD REMAINING LAYERS

            if index then
                for i=index,(#args.n_units-1) do
                  convLayers:add(convLayer(args.n_units[i], args.n_units[i+1],
                                      args.filter_size[i+1], args.filter_size[i+1],
                                      args.filter_stride[i+1], args.filter_stride[i+1]))
                  convLayers:add(args.nl())
                end
            end
      --==============================================================================

        else       -- CREATION OF A NEW EMPTY NETWORK
            --net = nn.Sequential()
            --net:add(nn.Reshape(unpack(args.input_dims)))
            convLayers:add(convLayer(args.hist_len*args.ncols, args.n_units[1],
                                args.filter_size[1], args.filter_size[1],
                                args.filter_stride[1], args.filter_stride[1],1))
            convLayers:add(args.nl())

            -- Add convolutional layers
            for i=1,(#args.n_units-1) do
                -- second convolutional layer
                convLayers:add(convLayer(args.n_units[i], args.n_units[i+1],
                                    args.filter_size[i+1], args.filter_size[i+1],
                                    args.filter_stride[i+1], args.filter_stride[i+1]))
                convLayers:add(args.nl())
            end
        end

        net:add(convLayers)
    end

    local nel
    if args.gpu >= 0 then
        if args.gpu_type == 1 then
            -- net:add(nn.Transpose({4,3},{3,2},{2,1})) -> deprecated with CUDA
            nel = net:cuda():forward(torch.zeros(1,unpack(args.input_dims))
                    :cuda()):nElement()
        else
          nel = net:cl():forward(torch.zeros(1,unpack(args.input_dims))
                  :cl()):nElement()
        end
    else
        nel = net:forward(torch.zeros(1,unpack(args.input_dims))):nElement()
    end

if args.only_conv ~= 1 then

    -- reshape all feature planes into a vector per example
    local netFC = nn.Sequential()
    netFC:add(nn.Reshape(nel))

    -- fully connected layer
    netFC:add(nn.Linear(nel, args.n_hid[1]))
    netFC:add(args.nl())
    local last_layer_size = args.n_hid[1]

    for i=1,(#args.n_hid-1) do
        -- add Linear layer
        last_layer_size = args.n_hid[i+1]
        netFC:add(nn.Linear(args.n_hid[i], last_layer_size))
        netFC:add(args.nl())
    end

    -- add the last fully connected layer (to actions)
    netFC:add(nn.Linear(last_layer_size, args.n_actions))

    net:add(netFC)

end


    if args.gpu >=0 then
        if args.gpu_type == 1 then
            net:cuda()
        else
            net:cl()
        end
    end
    if args.verbose >= 2 then
        print(net)
        print('Convolutional layers flattened output size:', nel)
    end

    return net
end
