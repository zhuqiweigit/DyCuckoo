#ifndef DATA_LAYOUT_H
#define DATA_LAYOUT_H
#include "../include/dynamic_helpers.cuh"
#include "../thirdParty/cnmem.h"
#include <limits>
#include <helper_cuda.h>

template<
        typename Entry = unsigned long long,
        typename Key = uint32_t,
        typename Value = uint32_t,
        uint8_t KeyBits = 32,
        uint8_t ValBits = 32,
        Key EmptyKey = 0,
        Value EmptyValue = 0,
        uint32_t BucketSize = 16,
        uint8_t TableNum = 4,
        uint32_t errorTableLen = 10000
        >
class DataLayout{
public:
    using key_t = Key;
    using value_t = Value;
    using entry_t = Entry;

    static constexpr uint8_t key_bits = KeyBits;
    static constexpr uint8_t val_bits = ValBits;
    static constexpr key_t empty_key = EmptyKey;
    static constexpr value_t empty_val = EmptyValue;
    static constexpr entry_t key_mask = (1UL << key_bits) - 1;
    static constexpr entry_t val_mask = ((1UL << val_bits) - 1) << key_bits;

    static const uint32_t bucket_size = BucketSize;

    static const uint32_t error_table_len = errorTableLen;

    static const uint8_t table_num = TableNum;

public:
    class data_t{
    public:
        entry_t entry;

        HOSTDEVICEQUALIFIER
        data_t(){
            entry = empty_val;
            entry = (entry << val_bits) + empty_key;
        }

        HOSTDEVICEQUALIFIER
        data_t(const entry_t& entry){
            this->entry = entry;
        }

        HOSTDEVICEQUALIFIER
        data_t(const key_t& key, const value_t& val){
            set_entry(key, val);
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        void set_entry(const key_t& key, const value_t val){
            entry = val;
            entry = (entry << val_bits) + key;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        key_t get_key(){
            return entry & key_mask;
        }

        HOSTDEVICEQUALIFIER INLINEQUALIFIER
        value_t get_value(){
            return (entry & val_mask) >> key_bits;
        }
    };
public:

    class bucket_t{
    public:
        data_t bucket_data[bucket_size];
    };
public:
    class cuckoo_t{
    public:
        bucket_t* table_group[table_num];
        //count bucket num in single table
        uint32_t table_size[table_num];

        HOSTQUALIFIER
        static void device_table_mem_init(cuckoo_t &mycuckoo, uint32_t single_table_size){
            for(uint32_t i = 0; i < table_num; i++){
                cnmemMalloc((void**) &(mycuckoo.table_group[i]), sizeof(bucket_t) * single_table_size, 0);
                cudaMemset(mycuckoo.table_group[i], 0, sizeof(bucket_t) * single_table_size);
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