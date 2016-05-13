package.path = package.path .. ";../dqn/?.lua"

require 'torch'
require 'nn'
require 'nngraph'
require 'cutorch'
require 'Rectifier'
cnv = require 'cnv1'


args = cnv()
print (args.n_units[1])
