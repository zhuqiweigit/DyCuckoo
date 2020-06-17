#ifndef HASH_FUNCTIONS_H
#define HASH_FUNCTIONS_H
#include "qualifiers.cuh"
namespace hashers{

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    hash1(uint32_t key) {
        key = ~key + (key << 15);
        key = key ^ (key >> 12);
        key = key + (key << 2);
        key = key ^ (key >> 4);
        key = key * 2057;
        key = key ^ (key >> 16);
        return (key);
    }

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    hash2(uint32_t a) {
        a = (a + 0x7ed55d16) + (a << 12);
        a = (a ^ 0xc761c23c) ^ (a >> 19);
        a = (a + 0x165667b1) + (a << 5);
        a = (a + 0xd3a2646c) ^ (a << 9);
        a = (a + 0xfd7046c5) + (a << 3);
        a = (a ^ 0xb55a4f09) ^ (a >> 16);
        return a;
    }

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    hash3(uint32_t sig) {
        return ((sig ^ 59064253) + 72355969) % PRIME_uint;
    }

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    hash4(uint32_t a) {
        a = (a ^ 61) ^ (a >> 16);
        a = a + (a << 3);
        a = a ^ (a >> 4);
        a = a * 0x27d4eb2d;
        a = a ^ (a >> 15);
        return a;
    }

    DEVICEQUALIFIER INLINEQUALIFIER uint32_t
    hash5(uint32_t a) {
        a -= (a << 6);
        a ^= (a >> 17);
        a -= (a << 9);
        a ^= (a << 4);
        a -= (a << 3);
        a ^= (a << 10);
        a ^= (a >> 15);
        return a;
    }

}

#endif
