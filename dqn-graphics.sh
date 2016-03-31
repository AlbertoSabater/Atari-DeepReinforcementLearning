#!/usr/bin/env bash

if [[ $1 == "-r" ]]; then 
  sed -i -e "46s/qlua/luajit/" run_cpu
  sed -i -e "52s/true/false/" torch/share/lua/5.1/alewrap/AleEnv.lua
  exit 0
fi

torch/bin/luarocks install qttorch

sed -i -e "46s/luajit/qlua/" run_cpu
sed -i -e "52s/false/true/" torch/share/lua/5.1/alewrap/AleEnv.lua

echo "Display is now enabled for run_cpu"
echo "Now, in case you haven't done it yet, copy your ATARI ROMs to the roms folder. Make sure the file name of the ROM is all lower case, otherwise you'll get an error.

