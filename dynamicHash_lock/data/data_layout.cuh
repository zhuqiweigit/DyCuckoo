#ifndef DATA_LAYOUT_H
#define DATA_LAYOUT_H
#include "../include/dynamic_helpers.cuh"
#include "../thirdParty/cnmem.h"
#include <limits>
#include <helper_cuda.h>

template<
        typename Key = uint32_t,
        uint32_t KeyBits = 32,
        uint32_t ValBits = 128,
        Key EmptyKey = 0,
        uint32_t EmptyValue = 0,
        uint32_t BucketSize = 16,
        uint32_t TableNum = 4,
        uint32_t errorTableLen = 10000,
        uint32_t lockTag = 1,
        uint32_t unlockTag = 0
        >
class DataLayout{
public:
    using key_t = Key;

    static constexpr uint32_t key_bits = KeyBits;
    static constexpr uint32_t val_bits = ValBits;
    static constexpr uint32_t val_lens = ValBits / 32;
    static constexpr key_t empty_key = EmptyKey;
    static constexpr uint32_t empty_val = EmptyValue;
    static constexpr uint32_t lock_tag = lockTag;
    static constexpr uint32_t unlock_tag = unlockTag;

    static const uint32_t bucket_size = BucketSize;

    static const uint32_t error_table_len = errorTableLen;

    static const uint32_t table_num = TableNum;

public:
    class value_t{
    public:
        uint32_t data[val_lens];

        HOSTDEVICEQUALIFIER
        value_t(){}

        HOSTDEVICEQUALIFIER
        value_t(const value_t& val){
            for(int i = 0; i < val_lens; i++)
                data[i] = val.data[i];
        }

        HOSTDEVICEQUALIFIER
        value_t& operator=(const value_t& val){
            for(int i = 0; i < val_lens; i++)
                data[i] = val.data[i];
            return *this;
        }

    };
public:

    class key_bucket_t{
    public:
        key_t bucket_data[bucket_size];
    };

    class value_bucket_t{
    public:
        value_t bucket_data[bucket_size];
    };

public:
    template<uint32_t ThreadNum = 512,
            uint32_t CgSize = 16>
    class Smem{
    public:
        value_t value_to_write[ThreadNum / CgSize];
    };

public:
    class cuckoo_t{
    public:
        key_bucket_t* key_table_group[table_num];
        value_bucket_t* value_table_group[table_num];
        //one bucket one lock
        uint32_t *bucket_lock[table_num];
        //count bucket num in single table
        uint32_t table_size[table_num];
        HOSTQUALIFIER
        static void device_table_mem_init(cuckoo_t &mycuckoo, uint32_t single_table_size){
            for(uint32_t i = 0; i < table_num; i++){
                cnmemMalloc((void**) &(mycuckoo.key_table_group[i]), sizeof(key_bucket_t) * single_table_size, 0);
                cnmemMalloc((void**) &(mycuckoo.value_table_group[i]), sizeof(value_bucket_t) * single_table_size, 0);
                cnmemMalloc((void**)&(mycuckoo.bucket_lock[i]), sizeof(uint32_t) * single_table_size, 0);
                cudaMemset(mycuckoo.key_table_group[i], 0, sizeof(key_bucket_t) * single_table_size);
                cudaMemset(mycuckoo.value_table_group[i], 0, sizeof(value_bucket_t) * single_table_size);
                cudaMemset(mycuckoo.bucket_lock[i], unlockTag, sizeof(uint32_t) * single_table_size);
                mycuckoo.table_size[i] = single_table_size;
            }
            checkCudaErrors(cudaGetLastError());
        }

    };

public:
    class error_table_t{
    public:
        key_t * error_keys;
        value_t* error_values;
        uint32_t error_pt;

        HOSTQUALIFIER
        void device_mem_init(){
            error_pt = 0;
            cnmemMalloc((void**)&error_keys, sizeof(key_t) * error_table_len, 0);
            cnmemMalloc((void**)&error_values, sizeof(value_t)* error_table_len, 0);
            checkCudaErrors(cudaGetLastError());
        }

    };
};

#endif