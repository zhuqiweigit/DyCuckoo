# DyCuckoo
Dycuckoo is a dynamic hash table implemented on GPU. 

## Environment
The code is tested on:
* Ubuntu 18.04
* NVIDIA Titan V GPU
* CUDA 11

## get started
* clone this repo
* mkdir build && cd build
* cmake .. && make
* for static test: ```./static_test [file_name] [key_len] [load_factor]```
* for dynamic test: ```./dynamic_test [file_name] [key_len] [r] [batch_size] [lower_bound] [upper_bound] [init_load_factor]```
