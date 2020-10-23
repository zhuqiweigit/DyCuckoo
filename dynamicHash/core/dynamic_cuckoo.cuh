#ifndef DYNAMIC_CUCKOO_H
#define DYNAMIC_CUCKOO_H
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
class DynamicCuckoo{
public:
    using data_t = DataLayout<>::data_t;
    using key_t = DataLayout<>::key_t;
    using value_t = DataLayout<>::value_t;
    using entry_t = DataLayout<>::entry_t;
    using bucket_t = DataLayout<>::bucket_t;
    using cuckoo_t = DataLayout<>::cuckoo_t;
    using error_table_t = DataLayout<>::error_table_t;
    static constexpr key_t empty_key = DataLayout<>::empty_key;
    static constexpr value_t empty_val = DataLayout<>::empty_val;
    static constexpr uint32_t bucket_size = DataLayout<>::bucket_size;
    static constexpr uint8_t table_num = DataLayout<>::table_num;

    static constexpr uint32_t thread_num = ThreadNum;
    static constexpr uint32_t block_num = BlockNum;

    const double lower_bound;
    const double upper_bound;
    const int small_batch_size;

    cuckoo_t *host_cuckoo_table;
    error_table_t* host_error_table;

    uint32_t all_table_capacity;
    uint64_t all_kv_num;

    cnmemDevice_t device;

    DynamicCuckoo(uint32_t init_kv_num,
                  int small_batch,
                  double lower,
                  double upper):lower_bound(lower), upper_bound(upper), small_batch_size(small_batch){
        memset(&device, 0, sizeof(device));
        device.size = (size_t)4 * 1024 * 1024 * 1024;
        cnmemInit(1, &device, CNMEM_FLAGS_DEFAULT);
        checkCudaErrors(cudaGetLastError());

        all_kv_num = 0;
        uint32_t s = init_kv_num / (table_num * bucket_size);
        s = ch::nextPrime(s);
        uint32_t s_bucket = (s & 1) ? s + 1: s;
        all_table_capacity = s_bucket * table_num;
        host_cuckoo_table = (cuckoo_t *) malloc(sizeof(cuckoo_t));
        cuckoo_t::device_table_mem_init(*host_cuckoo_table, s_bucket);
        DynamicHash::meta_data_to_device(*host_cuckoo_table);

        //error table
        host_error_table = new error_table_t;
        host_error_table->device_mem_init();
        DynamicHash::meta_data_to_device(*host_error_table);

    }
    ~DynamicCuckoo(){
        for(uint32_t i = 0; i < table_num; i++){
            cnmemFree(host_cuckoo_table->table_group[i], 0);
        }
        free(host_cuckoo_table);

        cnmemFree(host_error_table->error_keys, 0);
        cnmemFree(host_error_table->error_values, 0);
        free(host_error_table);

        cnmemRelease();
    }

    void resize_up(){
        uint32_t table_to_resize_no, min_size = std::numeric_limits<uint32_t>::max();
        for(uint32_t i = 0; i < table_num; i++){
            if(host_cuckoo_table->table_size[i] < min_size){
                table_to_resize_no = i;
                min_size = host_cuckoo_table->table_size[i];
            }
        }
        all_table_capacity += min_size;
        uint32_t old_size = min_size;
        uint32_t new_size = min_size * 2;
        bucket_t *new_table;

        cnmemMalloc((void**)&new_table, sizeof(bucket_t) * new_size, 0);
        checkCudaErrors(cudaGetLastError());
        cudaMemset((void**)new_table, 0, sizeof(bucket_t) * new_size);
        checkCudaErrors(cudaGetLastError());
        //update meta data
        bucket_t *old_table = host_cuckoo_table->table_group[table_to_resize_no];
        host_cuckoo_table->table_group[table_to_resize_no] = new_table;
        host_cuckoo_table->table_size[table_to_resize_no] = new_size;
        DynamicHash::meta_data_to_device(*host_cuckoo_table);
        //upsize
        DynamicHash::cuckoo_resize_up<<<block_num, thread_num>>>(old_table, old_size, table_to_resize_no);
        cnmemFree(old_table, 0);
    }


    void resize_down(){
        uint32_t table_to_resize_no, max_size = std::numeric_limits<uint32_t>::min();
        for(uint32_t i = 0; i < table_num; i++){
            if(host_cuckoo_table->table_size[i] > max_size){
                table_to_resize_no = i;
                max_size = host_cuckoo_table->table_size[i];
            }
        }
        uint32_t new_size = (max_size + 1) / 2;
        uint32_t old_size = max_size;
        all_table_capacity = all_table_capacity - (max_size - new_size);
        bucket_t *new_table;
        cnmemMalloc((void**)&new_table, sizeof(bucket_t) * new_size, 0);
        checkCudaErrors(cudaGetLastError());
        cudaMemset((void**)new_table, 0, sizeof(bucket_t) * new_size);
        checkCudaErrors(cudaGetLastError());
        //update meta data
        bucket_t *old_table = host_cuckoo_table->table_group[table_to_resize_no];
        host_cuckoo_table->table_group[table_to_resize_no] = new_table;
        host_cuckoo_table->table_size[table_to_resize_no] = new_size;
        DynamicHash::meta_data_to_device(*host_cuckoo_table);
        checkCudaErrors(cudaGetLastError());
        //down size
        DynamicHash::cuckoo_resize_down_pre<<<block_num, thread_num>>>(old_table, old_size, table_to_resize_no);
        DynamicHash::cuckoo_resize_down<<<block_num, thread_num>>>(old_table, old_size, table_to_resize_no);
        cnmemFree(old_table, 0);
    }

    INLINEQUALIFIER
    void test_resize(uint64_t kv_num_after_insert){
        while(kv_num_after_insert > (uint64_t)(upper_bound * (all_table_capacity * bucket_size))){
            resize_up();
        }
        if( kv_num_after_insert < (uint64_t)(lower_bound * (all_table_capacity * bucket_size)) && kv_num_after_insert > small_batch_size * 11){
            resize_down();
        }
    }

    void batch_insert(key_t *keys_d, value_t* values_d, uint32_t size){

        uint64_t after_insert_size = size + all_kv_num;
        test_resize(after_insert_size);
        all_kv_num += size;
        DynamicHash::cuckoo_insert<<< block_num, thread_num >>> (keys_d , values_d, size);

        /**
         * error handle: resize up and reinsert error data
         * */

       /* cudaMemcpyFromSymbol(host_error_table, error_table, sizeof(error_table_t));
        if(host_error_table->error_pt != 0){
            resize_up();
            DynamicHash::cuckoo_insert<<< block_num, thread_num >>> (error_table.error_keys , error_table.error_values, error_table.error_pt);
            cudaMemcpyFromSymbol(host_error_table, error_table, sizeof(error_table_t));
            host_error_table->error_pt = 0;
            DynamicHash::meta_data_to_device(*host_error_table);
        }*/
    }


    void batch_search(key_t *keys_d, value_t *values_d, uint32_t size){

        DynamicHash::cuckoo_search <<< block_num, thread_num >>> (keys_d, values_d, size);

    }


    void batch_delete(key_t *keys_d, value_t *values_d, uint32_t size){

        DynamicHash::cuckoo_delete<<< block_num, thread_num >>> (keys_d, values_d, size);
        uint64_t after_delete_size = all_kv_num - size;
        all_kv_num -= size;
        test_resize(after_delete_size);
    }


};



#endif