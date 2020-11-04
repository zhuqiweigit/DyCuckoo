# DyCuckoo
Source code for the paper [DyCuckoo: Dynamic Hash Tables on GPUs]()

Dycuckoo is a dynamic hash table implemented on GPU which achieves superior efficiency and enables fine-grained memory control.

This repository contains two version of Dycuckoo to deal with different key-value pair length. If the key-value pair is within 64 bits, we recommend to use [dynamicHash](./dynamicHash) to achieve higher performance. Otherwise you should use [dynamicHash_lock](./dynamicHash_lock) which can support larger key-value pair length. 

## Environment
The code is tested on:
* Ubuntu 18.04
* NVIDIA Titan V GPU
* CUDA 11

## get started
* clone this repo
* mkdir build && cd build
* cmake .. && make
* For static test: ```./static_test [file_name] [key_len] [load_factor]```
  * ```file_name```  is the path of the input file. The file only contains keys saved in binary format.
  * ```key_len``` is the number of keys in the input file.
  * ```load_factor``` is the filled factor of the hash table.
  * Example: ```./static_test ../data/ali-unique.dat 4583941 0.85```
* for dynamic test: ```./dynamic_test [file_name] [key_len] [r] [batch_size] [lower_bound] [upper_bound] [init_load_factor]```
  * ```file_name```  is the path of the input file. The file only contains keys saved in binary format.
  * ```key_len``` is the number of keys in the input file.
  * ```r``` is the ratio of deletions over insertions in a batch.(r = 3 means the ratio is 0.3)
  * ```batch_size``` is the number of kv pairs in a single batch.
  * ```lower_bound``` and ```upper_bound``` are parameters to control the filled factor when inserting and deleting kv pairs.
  * ```init_load_factor``` is the initial filled factor. 
  * Example: ```./dynamic_test ../data/random_unique.dat 100000000 2 1000000 0.5 0.85 0.85```

