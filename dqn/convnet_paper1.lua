--[[
Copyright (c) 2014 Google Inc.

See LICENSE file for full terms of limited license.
]]

-- DEFINICIÓN DE LAS CAPAS DE LA RED NEURONAL

require 'convnet'

return function(args)

    args.n_units        = {16, 32}
    args.filter_size    = {8, 4}
    args.filter_stride  = {4, 2}
    args.n_hid          = {256}
    args.nl             = nn.Rectifier

    return create_network(args)
end

