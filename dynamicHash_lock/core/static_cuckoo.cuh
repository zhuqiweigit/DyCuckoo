#ifndef STATIC_CUCKOO_H
#define STATIC_CUCKOO_H
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdint.h>
#include "../data/data_layout.cuh"
#include "dynamic_hash.cuh"
#include "../thirdParty/cnmem.h"
#include "../tools/gputimer.h"
using namespace cuckoo_helpers;
using namespace hashers;
using namespace DynamicHash;
namespace ch = cuckoo_helpers;
template<
        uint32_t ThreadNum = 512,
        uint32_t BlockNum = 512
>
class StaticCuckoo{
public:
    using key_t = DataLayout<>::key_t;
    using value_t = DataLayout<>::value_t;
    using key_bucket_t = DataLayout<>::key_bucket_t;
    using value_bucket_t = DataLayout<>::value_bucket_t;
    using cuckoo_t = DataLayout<>::cuckoo_t;
    using error_table_t = DataLayout<>::error_table_t;

    static constexpr key_t empty_key = DataLayout<>::empty_key;
    static constexpr uint32_t bucket_size = DataLayout<>::bucket_size;
    static constexpr uint32_t table_num = DataLayout<>::table_num;

    static constexpr uint32_t thread_num = ThreadNum;
    static constexpr uint32_t block_num = BlockNum;

    cuckoo_t *host_cuckoo_table;
    error_table_t* host_error_table;

    cnmemDevice_t device;


    StaticCuckoo(uint32_t init_kv_num){
        ///cnmem init
        memset(&device, 0, sizeof(device));
        device.size = (size_t)4*1024*1024*1024; /// more =(size_t) (0.95*props.totalGlobalMem);
        cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
        checkCudaErrors(cudaGetLastError());

        uint32_t s = init_kv_num / (table_num * bucket_size);
        //bucket num in a table
        s = ch::nextPrime(s);
        uint32_t s_bucket = (s & 1) ? s + 1: s;
        host_cuckoo_table = (cuckoo_t *) malloc(sizeof(cuckoo_t));
        cuckoo_t::device_table_mem_init(*host_cuckoo_table, s_bucket);
        checkCudaErrors(cudaGetLastError());
        DynamicHash::meta_data_to_device(*host_cuckoo_table);


        //error table
        host_error_table = new error_table_t;
        host_error_table->device_mem_init();
        DynamicHash::meta_data_to_device(*host_error_table);

    }
    ~StaticCuckoo(){
        for(uint32_t i = 0; i < table_num; i++){
            cnmemFree(host_cuckoo_table->key_table_group[i], 0);
            cnmemFree(host_cuckoo_table->value_table_group[i], 0);
            cnmemFree(host_cuckoo_table->bucket_lock[i], 0);
        }
        free(host_cuckoo_table);
        cnmemFree(host_error_table->error_keys, 0);
        cnmemFree(host_error_table->error_values, 0);
        free(host_error_table);

        cnmemRelease();
    }

    void hash_insert(key_t *keys, value_t* values, uint32_t size){

        key_t *dev_keys;
        value_t* dev_values;
        cnmemMalloc((void**)&dev_keys, sizeof(key_t) * size, 0);
        cudaMemcpy(dev_keys, keys, sizeof(key_t) * size, cudaMemcpyHostToDevice);
        cnmemMalloc((void**)&dev_values, sizeof(value_t) * size, 0);
        cudaMemcpy(dev_values, values, sizeof(value_t) * size, cudaMemcpyHostToDevice);

        GpuTimer timer;
        timer.Start();
        DynamicHash::cuckoo_insert<<< block_num, thread_num >>> (dev_keys, dev_values, size);
        timer.Stop();
        double diff = timer.Elapsed()* 1000000;
        printf("<insert> %.2f\n", (double) (size) / diff);

        cnmemFree(dev_keys, 0);
        cnmemFree(dev_values, 0);
        checkCudaErrors(cudaGetLastError());
    }

    void hash_search(key_t *keys, value_t *values, uint32_t size){

        key_t *dev_keys;
        value_t* dev_values;
        cnmemMalloc((void**)&dev_keys, sizeof(key_t) * size, 0);
        cudaMemcpy(dev_keys, keys, sizeof(key_t) * size, cudaMemcpyHostToDevice);
        cnmemMalloc((void**)&dev_values, sizeof(value_t) * size, 0);
        cudaMemset(dev_values,0, sizeof(value_t) * size);

        GpuTimer timer;
        timer.Start();
        DynamicHash::cuckoo_search <<< block_num, thread_num >>> (dev_keys, dev_values, size);
        timer.Stop();
        double diff = timer.Elapsed()* 1000000;
        printf("<search> %.2f\n", (double) (size) / diff);

        cudaMemcpy(values, dev_values, sizeof(value_t) * size, cudaMemcpyDeviceToHost);
        cnmemFree(dev_keys, 0);
        cnmemFree(dev_values, 0);
        checkCudaErrors(cudaGetLastError());

    }

    void hash_delete(key_t *keys, value_t *values, uint32_t size){
        key_t *dev_keys;
        value_t* dev_values;
        cnmemMalloc((void**)&dev_keys, sizeof(key_t) * size, 0);
        cudaMemcpy(dev_keys, keys, sizeof(key_t) * size, cudaMemcpyHostToDevice);
        cnmemMalloc((void**)&dev_values, sizeof(value_t) * size, 0);
        cudaMemset(dev_values,0, sizeof(value_t) * size);

        DynamicHash::cuckoo_delete<<< block_num, thread_num >>> (dev_keys, dev_values, size);

        cnmemFree(dev_keys, 0);
        cnmemFree(dev_values, 0);
        checkCudaErrors(cudaGetLastError());
    }


};



#endif