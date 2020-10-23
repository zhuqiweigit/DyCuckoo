#ifndef DYNAMIC_HELPERS_H
#define DYNAMIC_HELPERS_H
#include "hash_functions.cuh"
#include "qualifiers.cuh"
#include <cstdint>

namespace cuckoo_helpers{

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    get_pair(uint32_t key){
        return (hashers::hash5(key) % 6);
    }

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    get_table1_no(uint32_t pair){
        if (pair & 1)
            return 0;
        if (pair & 2)
            return 2;
        return 1;
    }

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    get_table2_no(uint32_t pair){
        if (pair & 1)
        return (pair >> 1) + 1;
        if (pair & 2)
        return 3;
        return (pair >> 2) + 2;
    }

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    caculate_hash(uint32_t k, uint32_t table_no, uint32_t table_len){
        switch(table_no % 4){
            case 0:
                return hashers::hash1(k) % PRIME_uint % table_len;
            case 1:
                return hashers::hash2(k) % PRIME_uint % table_len;
            case 2:
                return hashers::hash3(k) % PRIME_uint % table_len;
            case 3:
                return hashers::hash4(k) % PRIME_uint % table_len;
        }
        return 0;
    }

    HOSTDEVICEQUALIFIER
    bool isPrime(int num)
    {
        if (num == 2 || num == 3) return true;
        if (num % 6 != 1 && num % 6 != 5) return false;
        for (int i = 5; i*i <= num; i += 6)
            if (num % i == 0 || num % (i+2) == 0) return false;
        return true;
    }

    HOSTDEVICEQUALIFIER
    uint32_t nextPrime(uint32_t n)
    {
        bool state=isPrime(n);
        while(!state)
            state=isPrime(++n);
        return n;
    }


}



#endif
