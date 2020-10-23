#ifndef DYNAMIC_HASH_H
#define DYNAMIC_HASH_H
#include <cooperative_groups.h>
#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdint.h>
#include "../data/data_layout.cuh"
#include "../thirdParty/cnmem.h"
using namespace cooperative_groups;
using namespace cuckoo_helpers;
using namespace hashers;

namespace cg = cooperative_groups;
namespace ch = cuckoo_helpers;

__constant__ DataLayout<>::cuckoo_t cuckoo_table;
__device__ DataLayout<>::error_table_t error_table;

namespace DynamicHash{

    using data_t = DataLayout<>::data_t;
    using key_t = DataLayout<>::key_t;
    using value_t = DataLayout<>::value_t;
    using entry_t = DataLayout<>::entry_t;
    using bucket_t = DataLayout<>::bucket_t;
    using cuckoo_t = DataLayout<>::cuckoo_t;
    using error_table_t = DataLayout<>::error_table_t;

    static constexpr uint8_t CgSize = 16;
    static constexpr uint8_t MaxEvictNum = 100;

    static constexpr key_t empty_key = DataLayout<>::empty_key;
    static constexpr value_t empty_val = DataLayout<>::empty_val;
    static constexpr uint32_t bucket_size = DataLayout<>::bucket_size;
    static constexpr uint32_t error_table_len = DataLayout<>::error_table_len;

    static constexpr uint8_t cg_size = CgSize;
    static constexpr uint8_t max_evict_num = MaxEvictNum;

    DEVICEQUALIFIER
    void cg_error_handle(data_t& data, thread_block_tile<cg_size> group){
        if(group.thread_rank() != 0)
            return;
        uint32_t ptr = atomicAdd(&(error_table.error_pt), 1);
        if(ptr >= error_table_len){
            return;
        }
        error_table.error_keys[ptr] = data.get_key();
        error_table.error_values[ptr] = data.get_value();
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    bool cg_inert(data_t &data, uint8_t pre_table_no, thread_block_tile<cg_size> group){
        auto lane_id = group.thread_rank();
        key_t key;
        uint32_t pair;
        uint32_t insert_table_no;
        for(uint32_t i = 0; i < MaxEvictNum; ++i){
            key = data.get_key();
            pair = ch::get_pair((uint32_t)key);
            insert_table_no = ch::get_table1_no(pair);
            if(insert_table_no == pre_table_no){
                insert_table_no = ch::get_table2_no(pair);
            }
            uint32_t table_len = cuckoo_table.table_size[insert_table_no];
            uint32_t hash_val = ch::caculate_hash((uint32_t)key, insert_table_no, table_len);
            bucket_t *bucket = cuckoo_table.table_group[insert_table_no] + hash_val;
            uint32_t stride = bucket_size / cg_size;
            for(uint32_t ptr = 0; ptr < stride; ptr++){
                data_t probe_data = (bucket->bucket_data)[ptr * cg_size + lane_id];
                key_t probe_key = probe_data.get_key();
                uint32_t group_mask =  group.ballot(probe_key == empty_key);
                bool success = false;
                while(group_mask != 0){
                    uint32_t group_leader = __ffs(group_mask) - 1;
                    if(lane_id == group_leader){
                        auto result = atomicCAS((entry_t *)(bucket->bucket_data) + ptr * cg_size + lane_id, probe_data.entry, data.entry);
                        if(result == probe_data.entry){
                            success = true;
                        }
                    }
                    if(group.any(success == true)){
                        return true;
                    }else{
                        probe_data = (bucket->bucket_data)[ptr * cg_size + lane_id];
                        probe_key = probe_data.get_key();
                        group_mask =  group.ballot(probe_key == empty_key);
                    }
                }
            }
            // probe fail, evict
            entry_t cas_result;
            if(lane_id == 0){
                cas_result = atomicExch((entry_t *)(bucket->bucket_data) + (i % cg_size), data.entry);
            }
            data.entry = group.shfl(cas_result, 0);
            pre_table_no = insert_table_no;
        }
        // insert fail, handle error
        cg_error_handle(data, group);
        return false;
    }


    GLOBALQUALIFIER
    void cuckoo_insert(key_t *keys, value_t *values, uint32_t data_num){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;

        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);

        for(; group_index_in_all < data_num; group_index_in_all += step){
            uint32_t pre_table_no;
            key_t key = keys[group_index_in_all];
            value_t value = values[group_index_in_all];
            uint32_t pair = ch::get_pair((uint32_t)key);
            if(key & 1){
                /// last bit == 1 , first insert to pos 2
                pre_table_no = ch::get_table1_no(pair);
            }else{
                pre_table_no = ch::get_table2_no(pair);
            }
            data_t data(key, value);
            cg_inert(data, pre_table_no, group);
        }
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    uint8_t cg_search_in_bucket(const key_t &key, value_t &val, bucket_t* bucket, thread_block_tile<cg_size> group){
        auto lane_id = group.thread_rank();
        uint32_t stride = bucket_size / cg_size;
        for(uint32_t ptr = 0; ptr < stride; ptr++){
            data_t probe_data = (bucket->bucket_data)[ptr * cg_size + lane_id];
            key_t probe_key = probe_data.get_key();
            uint32_t group_mask =  group.ballot(probe_key == key);
            if(group_mask != 0){
                uint32_t group_leader = __ffs(group_mask) - 1;
                if(lane_id == group_leader){
                    val = probe_data.get_value();
                }
                return true;
            }
        }
        return false;
    }

    GLOBALQUALIFIER
    void cuckoo_search(key_t* keys, value_t* values, uint32_t size){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);

        uint32_t pair, search_table_no, table_len, hash_val;
        bucket_t *bucket;
        uint8_t flag = false;
        for(;group_index_in_all < size; group_index_in_all += step){
            key_t key = keys[group_index_in_all];
            pair = ch::get_pair((uint32_t)key);
            if(key & 1) {
                search_table_no = ch::get_table2_no(pair);
            }else{
                search_table_no = ch::get_table1_no(pair);
            }
            table_len = cuckoo_table.table_size[search_table_no];
            hash_val = ch::caculate_hash((uint32_t)key, search_table_no, table_len);
            bucket = cuckoo_table.table_group[search_table_no] + hash_val;
            flag = cg_search_in_bucket(key, values[group_index_in_all], bucket, group);
            if(group.any(flag == true)){
               continue;
            }
            if(key & 1) {
                search_table_no = ch::get_table1_no(pair);
            }else{
                search_table_no = ch::get_table2_no(pair);
            }
            table_len = cuckoo_table.table_size[search_table_no];
            hash_val = ch::caculate_hash((uint32_t)key, search_table_no, table_len);
            bucket = cuckoo_table.table_group[search_table_no] + hash_val;
            flag = cg_search_in_bucket(key, values[group_index_in_all], bucket, group);
        }

    }

    DEVICEQUALIFIER INLINEQUALIFIER
    uint8_t cg_delete_in_bucket(const key_t &key, bucket_t* bucket, thread_block_tile<cg_size> group){
        auto lane_id = group.thread_rank();
        uint32_t stride = bucket_size / cg_size;
        for(uint32_t ptr = 0; ptr < stride; ptr++){
            data_t probe_data = (bucket->bucket_data)[ptr * cg_size + lane_id];
            key_t probe_key = probe_data.get_key();
            uint32_t group_mask = group.ballot(probe_key == key);
            if(group_mask != 0){
                uint32_t group_leader = __ffs(group_mask) - 1;
                if(lane_id == group_leader){
                    data_t empty_data;
                    (bucket->bucket_data)[ptr * cg_size + lane_id] = empty_data;
                }
                return true;
            }
        }
        return false;
    }

    GLOBALQUALIFIER
    void cuckoo_delete(key_t* keys, value_t* values, uint32_t size){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);

        uint32_t pair, delete_table_no, table_len, hash_val;
        bucket_t *bucket;
        uint8_t flag = false;
        for(;group_index_in_all < size; group_index_in_all += step){
            key_t key = keys[group_index_in_all];
            pair = ch::get_pair((uint32_t)key);
            if(key & 1) {
                delete_table_no = ch::get_table2_no(pair);
            }else{
                delete_table_no = ch::get_table1_no(pair);
            }
            table_len = cuckoo_table.table_size[delete_table_no];
            hash_val = ch::caculate_hash((uint32_t)key, delete_table_no, table_len);
            bucket = cuckoo_table.table_group[delete_table_no] + hash_val;
            flag = cg_delete_in_bucket(key, bucket, group);
            if(group.any(flag == true)){
                continue;
            }
            if(key & 1) {
                delete_table_no = ch::get_table1_no(pair);
            }else{
                delete_table_no = ch::get_table2_no(pair);
            }
            table_len = cuckoo_table.table_size[delete_table_no];
            hash_val = ch::caculate_hash((uint32_t)key, delete_table_no, table_len);
            bucket = cuckoo_table.table_group[delete_table_no] + hash_val;
            flag = cg_delete_in_bucket(key, bucket, group);
        }
    }

    /**
     * new table has been set to cuckoo table; and old table has been replaced;
     * */
    GLOBALQUALIFIER
    void cuckoo_resize_up(bucket_t *old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        auto lane_id = group.thread_rank();

        bucket_t *new_table = cuckoo_table.table_group[table_to_resize_no];
        uint32_t new_table_len = cuckoo_table.table_size[table_to_resize_no];
        for(; group_index_in_all < old_table_bucket_num; group_index_in_all += step){
            bucket_t *bucket = old_table + group_index_in_all;
            for(int32_t ptr = lane_id; ptr < bucket_size; ptr += cg_size){
                data_t data = (bucket->bucket_data)[ptr];
                key_t key = data.get_key();
                if(key != empty_key){
                    uint32_t new_hash_val = ch::caculate_hash((uint32_t)key, table_to_resize_no, new_table_len);
                    //todo: maybe a bug
                    new_table[new_hash_val].bucket_data[ptr] = data;
                }
            }
        }

    }

    /**
     * resize down: copy first half of old table to new table
     * */
    GLOBALQUALIFIER
    void cuckoo_resize_down_pre(bucket_t *old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        auto lane_id = group.thread_rank();
        bucket_t *new_table = cuckoo_table.table_group[table_to_resize_no];
        uint32_t new_table_bucket_num = cuckoo_table.table_size[table_to_resize_no];
        uint32_t ptr = group_index_in_all;
        bucket_t* old_bucket;
        bucket_t* new_bucket;
        for(; ptr < new_table_bucket_num; ptr += step){
            old_bucket = old_table + ptr;
            new_bucket = new_table + ptr;
            for(int32_t cg_ptr = lane_id; cg_ptr < bucket_size; cg_ptr += cg_size){
                (new_bucket->bucket_data)[cg_ptr] = (old_bucket->bucket_data)[cg_ptr];
            }
        }
    }

    /**
     * new table has been set to cuckoo table; and old table has been replaced;
     * */
    GLOBALQUALIFIER
    void cuckoo_resize_down(bucket_t *old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        auto lane_id = group.thread_rank();

        uint32_t new_table_bucket_num = cuckoo_table.table_size[table_to_resize_no];
        uint32_t ptr;
        bucket_t* old_bucket;

        for(ptr = group_index_in_all + new_table_bucket_num; ptr < old_table_bucket_num; ptr += step){
            old_bucket = old_table + ptr;
            for(int32_t cg_ptr = lane_id; cg_ptr < bucket_size; cg_ptr += cg_size){
                uint8_t active = 0;
                data_t probe_data = (old_bucket->bucket_data)[cg_ptr];
                key_t probe_key = probe_data.get_key();
                if(probe_key != empty_key){
                    active = 1;
                }
                auto group_mask = group.ballot(active == 1);
                while(group_mask != 0){
                    auto leader = __ffs(group_mask) - 1;
                    data_t insert_data;
                    key_t insert_key;
                    insert_data.entry = group.shfl(probe_data.entry, leader);
                    insert_key = insert_data.get_key();
                    uint32_t insert_pre_table_no, pair;
                    pair = ch::get_pair((uint32_t)insert_key);
                    if(insert_key & 1) {
                        insert_pre_table_no = ch::get_table1_no(pair);
                    }else{
                        insert_pre_table_no = ch::get_table2_no(pair);
                    }
                    cg_inert(insert_data, insert_pre_table_no, group);
                    if(lane_id == leader) active = 0;
                    group_mask = group.ballot(active == 1);
                }
            }
        }
    }


    HOSTQUALIFIER INLINEQUALIFIER
    void meta_data_to_device(cuckoo_t &host_ptr){
        cudaMemcpyToSymbol(cuckoo_table, &host_ptr, sizeof(cuckoo_t));
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }

    HOSTQUALIFIER INLINEQUALIFIER
    void meta_data_to_device(error_table_t& host_ptr){
        cudaMemcpyToSymbol(error_table, &host_ptr, sizeof(error_table_t));
        cudaDeviceSynchronize();
        checkCudaErrors(cudaGetLastError());
    }
};

#endif