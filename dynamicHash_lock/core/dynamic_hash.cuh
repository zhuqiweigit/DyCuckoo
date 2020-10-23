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

    using key_t = DataLayout<>::key_t;
    using value_t = DataLayout<>::value_t;
    using key_bucket_t = DataLayout<>::key_bucket_t;
    using value_bucket_t = DataLayout<>::value_bucket_t;
    using cuckoo_t = DataLayout<>::cuckoo_t;
    using error_table_t = DataLayout<>::error_table_t;
    using smem_t = DataLayout<>::Smem<>;

    static constexpr uint32_t CgSize = 16;
    static constexpr uint32_t MaxEvictNum = 100;

    static constexpr key_t empty_key = DataLayout<>::empty_key;
    static constexpr uint32_t val_lens = DataLayout<>::val_lens;

    static constexpr uint32_t bucket_size = DataLayout<>::bucket_size;
    static constexpr uint32_t error_table_len = DataLayout<>::error_table_len;
    static constexpr uint32_t lock_tag = DataLayout<>::lock_tag;
    static constexpr uint32_t unlock_tag = DataLayout<>::unlock_tag;

    static constexpr uint32_t cg_size = CgSize;
    static constexpr uint32_t max_evict_num = MaxEvictNum;

    __shared__ DataLayout<>::Smem<> smem;

    DEVICEQUALIFIER
    void cg_error_handle(key_t& key, value_t &value, thread_block_tile<cg_size> group){
        if(group.thread_rank() != 0)
            return;
        uint32_t ptr = atomicAdd(&(error_table.error_pt), 1);
        if(ptr >= error_table_len){
            return;
        }
        error_table.error_keys[ptr] = key;
        error_table.error_values[ptr] = value;
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    void cg_insert_with_lock(key_t &key, value_t value, uint32_t pre_table_no, bool active, thread_block_tile<cg_size> group){
        uint32_t pair = ch::get_pair((uint32_t)key);
        uint32_t insert_table_no = ch::get_table1_no(pair);
        if(insert_table_no == pre_table_no){
            insert_table_no = ch::get_table2_no(pair);
        }
        uint32_t evict_num = 0;
        uint32_t table_len = cuckoo_table.table_size[insert_table_no];
        uint32_t hash_val = ch::caculate_hash((uint32_t)key, insert_table_no, table_len);
        uint32_t *bucket_lock_addr = cuckoo_table.bucket_lock[insert_table_no] + hash_val;
        while(group.any(active == true)){
            __threadfence();
            bool lock_success = false;
            uint32_t group_mask = group.ballot(active == true);
            uint32_t leader = __ffs(group_mask) - 1;
            if(group.thread_rank() == leader){
                //check evict
                if(evict_num > MaxEvictNum){
                    active = false;
                    lock_success = false;
                }else{
                    auto result = atomicCAS(bucket_lock_addr, unlock_tag, lock_tag);
                    if(result == unlock_tag){
                        lock_success = true;
                    }
                }
            }
            //lock fail
            if(group.all(lock_success == false)){
                continue;
            }
            //broadcask insert_data
            uint32_t table_no = group.shfl(insert_table_no, leader);
            uint32_t insert_hash = group.shfl(hash_val, leader);
            key_bucket_t* key_insert_bucket = cuckoo_table.key_table_group[table_no] + insert_hash;
            value_bucket_t *value_bucket = cuckoo_table.value_table_group[table_no] + insert_hash;
            key_t insert_key = group.shfl(key, leader);
            key_t probe_key = (key_insert_bucket->bucket_data)[group.thread_rank()];
            uint32_t group_mask2 = group.ballot(probe_key == empty_key);
            //write data to smem
            if(group.thread_rank() == leader){
                smem.value_to_write[threadIdx.x / CgSize] = value;
            }
            if(group_mask2 != 0){
                uint32_t insert_thread = __ffs(group_mask2) - 1;
                if(group.thread_rank() == insert_thread){
                    (key_insert_bucket->bucket_data)[insert_thread] = insert_key;
                    (value_bucket->bucket_data)[insert_thread] = smem.value_to_write[threadIdx.x / CgSize];
                }
                group.sync();
                //unlock
                if(group.thread_rank() == leader){
                    atomicExch(bucket_lock_addr, unlock_tag);
                    active = false;
                }
            }else{
                //evict
                if(group.thread_rank() == leader){
                    key = (key_insert_bucket->bucket_data)[leader];
                    (key_insert_bucket->bucket_data)[leader] = insert_key;
                    value = (value_bucket->bucket_data)[leader];
                    (value_bucket->bucket_data)[leader] = smem.value_to_write[threadIdx.x / CgSize];
                    //unlock
                    atomicExch(bucket_lock_addr, unlock_tag);
                    pair = ch::get_pair((uint32_t)key);
                    uint32_t temp_table_no = ch::get_table1_no(pair);
                    if(temp_table_no == insert_table_no){
                        temp_table_no = ch::get_table2_no(pair);
                    }
                    insert_table_no = temp_table_no;
                    table_len = cuckoo_table.table_size[insert_table_no];
                    hash_val = ch::caculate_hash((uint32_t)key, insert_table_no, table_len);
                    bucket_lock_addr = cuckoo_table.bucket_lock[insert_table_no] + hash_val;
                    evict_num++;
                }
            }
        }

    }

    GLOBALQUALIFIER
    void cuckoo_insert(key_t *keys, value_t *values, uint32_t data_num){
        uint32_t step = gridDim.x * blockDim.x;
        uint32_t tid = blockDim.x * blockIdx.x + threadIdx.x;

        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        for(; tid < data_num ; tid += step){
            uint32_t pre_table_no;
            key_t key = keys[tid];
            uint32_t pair = ch::get_pair((uint32_t)key);
            if(key & 1){
                /// last bit == 1 , first insert to pos 2
                pre_table_no = ch::get_table1_no(pair);
            }else{
                pre_table_no = ch::get_table2_no(pair);
            }
            cg_insert_with_lock(key, values[tid], pre_table_no, true, group);
        }
    }

    DEVICEQUALIFIER INLINEQUALIFIER
    bool cg_search_in_bucket(const key_t &key, value_t &val, key_bucket_t* bucket, value_bucket_t * value_bucket, thread_block_tile<cg_size> group){
        auto lane_id = group.thread_rank();
        uint32_t stride = bucket_size / cg_size;
        for(uint32_t ptr = 0; ptr < stride; ptr++){
            key_t probe_key = (bucket->bucket_data)[ptr * cg_size + lane_id];
            uint32_t group_mask =  group.ballot(probe_key == key);
            if(group_mask != 0){
                uint32_t group_leader = __ffs(group_mask) - 1;
                if(lane_id == group_leader){
                    val = (value_bucket->bucket_data)[ptr * cg_size + lane_id];
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
        key_bucket_t *key_bucket;
        value_bucket_t *value_bucket;
        bool flag = false;
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
            key_bucket = cuckoo_table.key_table_group[search_table_no] + hash_val;
            value_bucket = cuckoo_table.value_table_group[search_table_no] + hash_val;
            flag = cg_search_in_bucket(key, values[group_index_in_all], key_bucket, value_bucket, group);
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
            key_bucket = cuckoo_table.key_table_group[search_table_no] + hash_val;
            value_bucket = cuckoo_table.value_table_group[search_table_no] + hash_val;
            flag = cg_search_in_bucket(key, values[group_index_in_all], key_bucket, value_bucket, group);
        }

    }

    DEVICEQUALIFIER INLINEQUALIFIER
    bool cg_delete_in_bucket(const key_t &key, key_bucket_t* key_bucket, thread_block_tile<cg_size> group){
        auto lane_id = group.thread_rank();
        uint32_t stride = bucket_size / cg_size;
        for(uint32_t ptr = 0; ptr < stride; ptr++){
            key_t probe_key = (key_bucket->bucket_data)[ptr * cg_size + lane_id];
            uint32_t group_mask = group.ballot(probe_key == key);
            if(group_mask != 0){
                uint32_t group_leader = __ffs(group_mask) - 1;
                if(lane_id == group_leader){
                    (key_bucket->bucket_data)[ptr * cg_size + lane_id] = empty_key;
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
        key_bucket_t *key_bucket;
        bool flag = false;
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
            key_bucket = cuckoo_table.key_table_group[delete_table_no] + hash_val;
            flag = cg_delete_in_bucket(key, key_bucket, group);
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
            key_bucket = cuckoo_table.key_table_group[delete_table_no] + hash_val;
            flag = cg_delete_in_bucket(key, key_bucket, group);
        }
    }

    /**
     * new table has been set to cuckoo table; and old table has been replaced;
     * */
    GLOBALQUALIFIER
    void cuckoo_resize_up(key_bucket_t *key_old_table, value_bucket_t * value_old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        auto lane_id = group.thread_rank();

        key_bucket_t *key_new_table = cuckoo_table.key_table_group[table_to_resize_no];
        value_bucket_t *value_new_table = cuckoo_table.value_table_group[table_to_resize_no];
        uint32_t new_table_len = cuckoo_table.table_size[table_to_resize_no];
        for(; group_index_in_all < old_table_bucket_num; group_index_in_all += step){
            key_bucket_t *key_bucket = key_old_table + group_index_in_all;
            value_bucket_t *value_bucket = value_old_table + group_index_in_all;
            for(int32_t ptr = lane_id; ptr < bucket_size; ptr += cg_size){
                key_t key = (key_bucket->bucket_data)[ptr];
                if(key != empty_key){
                    uint32_t new_hash_val = ch::caculate_hash((uint32_t)key, table_to_resize_no, new_table_len);
                    //todo: maybe a bug
                    key_new_table[new_hash_val].bucket_data[ptr] = key;
                    value_new_table[new_hash_val].bucket_data[ptr] = (value_bucket->bucket_data)[ptr];
                }
            }
        }

    }

    /**
     * resize down: copy first half of old table to new table
     * */
    GLOBALQUALIFIER
    void cuckoo_resize_down_pre(key_bucket_t *key_old_table, value_bucket_t *value_old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);
        auto lane_id = group.thread_rank();
        key_bucket_t *key_new_table = cuckoo_table.key_table_group[table_to_resize_no];
        value_bucket_t *value_new_table = cuckoo_table.value_table_group[table_to_resize_no];
        uint32_t new_table_bucket_num = cuckoo_table.table_size[table_to_resize_no];
        uint32_t ptr = group_index_in_all;
        key_bucket_t* key_old_bucket, *key_new_bucket;
        value_bucket_t *value_old_bucket, *value_new_bucket;
        for(; ptr < new_table_bucket_num; ptr += step){
            key_old_bucket = key_old_table + ptr;
            key_new_bucket = key_new_table + ptr;
            value_old_bucket = value_old_table + ptr;
            value_new_bucket = value_new_table + ptr;
            for(int32_t cg_ptr = lane_id; cg_ptr < bucket_size; cg_ptr += cg_size){
                (key_new_bucket->bucket_data)[cg_ptr] = (key_old_bucket->bucket_data)[cg_ptr];
                (value_new_bucket->bucket_data)[cg_ptr] = (value_old_bucket->bucket_data)[cg_ptr];
            }
        }
    }

    /**
     * new table has been set to cuckoo table; and old table has been replaced;
     * */
    GLOBALQUALIFIER
    void cuckoo_resize_down(key_bucket_t *key_old_table, value_bucket_t *value_old_table, uint32_t old_table_bucket_num, uint32_t table_to_resize_no){
        uint32_t step = (gridDim.x * blockDim.x) / cg_size;
        uint32_t group_index_in_all = (blockDim.x * blockIdx.x + threadIdx.x) / cg_size;
        auto block = cg::this_thread_block();
        auto group = cg::tiled_partition<cg_size>(block);

        uint32_t new_table_bucket_num = cuckoo_table.table_size[table_to_resize_no];
        uint32_t ptr;
        key_bucket_t* key_old_bucket;
        value_bucket_t *value_old_bucket;
        for(ptr = group_index_in_all + new_table_bucket_num; ptr < old_table_bucket_num; ptr += step){
            key_old_bucket = key_old_table + ptr;
            value_old_bucket = value_old_table + ptr;
            bool active = true;
            key_t probe_key = (key_old_bucket->bucket_data)[group.thread_rank()];
            if(probe_key == empty_key){
                active = false;
            }
            uint32_t insert_pre_table_no, pair;
            pair = ch::get_pair((uint32_t)probe_key);
            if(probe_key & 1) {
                insert_pre_table_no = ch::get_table1_no(pair);
            }else{
                insert_pre_table_no = ch::get_table2_no(pair);
            }
            cg_insert_with_lock(probe_key, (value_old_bucket->bucket_data)[group.thread_rank()], insert_pre_table_no, active, group);
        }
    }


    HOSTQUALIFIER INLINEQUALIFIER
    void meta_data_to_device(cuckoo_t &host_ptr){
        cudaMemcpyToSymbol(cuckoo_table, &host_ptr, sizeof(cuckoo_t));
    }

    HOSTQUALIFIER INLINEQUALIFIER
    void meta_data_to_device(error_table_t& host_ptr){
        cudaMemcpyToSymbol(error_table, &host_ptr, sizeof(error_table_t));
    }
};

#endif