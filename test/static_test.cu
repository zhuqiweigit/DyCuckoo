#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdint.h>
#include "../tools/gputimer.h"
#include "../data/data_layout.cuh"
#include "../core/static_cuckoo.cuh"
namespace ch = cuckoo_helpers;
using namespace std;

class StaticTest{
public:
    using data_t = DataLayout<>::data_t;
    using key_t = DataLayout<>::key_t;
    using value_t = DataLayout<>::value_t;
    static key_t *read_data(char *file_name, int data_len) {
        FILE *fid;
        fid = fopen(file_name, "rb");
        key_t *pos = (key_t *) malloc(sizeof(key_t) * data_len);
        if (fid == NULL) {
            printf("file not found.\n");
            return pos;
        }
        fread(pos, sizeof(unsigned int), data_len, fid);
        fclose(fid);
        return pos;
    }

    static void check(value_t *check_pool_h, int32_t size) {
        uint32_t error_cnt = 0;
        for (int i = 0; i < size; i++) {
            if (check_pool_h[i] != i + 5) {
                ++error_cnt;
            }
        }
        if (error_cnt != 0) {
            printf("num error:%d \n", error_cnt);
        } else {
            printf("batch check ok\n");
        }
    }

};
int main(int argc, char** argv) {
    using test_t = StaticTest;
    if (argc < 4)
    {
        cout << "para error\n" << endl;
        return -1;
    }

    char* file_name = argv[1];
    int pool_len = atoi(argv[2]);
    double init_fill_factor = atof(argv[3]);

    test_t::key_t* keys_h = test_t::read_data(file_name, pool_len);
    test_t::value_t *values_h = new test_t::value_t [pool_len], *check_h = new test_t::value_t [pool_len];
    for(int i = 0; i < pool_len; i++){
        values_h[i] = i + 5;
        check_h[i] = 0;
    }

    StaticCuckoo<512, 512> static_cuckoo(pool_len / init_fill_factor);
    static_cuckoo.hash_insert(keys_h, values_h, pool_len);
    static_cuckoo.hash_search(keys_h, check_h, pool_len);
    test_t::check(check_h, pool_len);

    delete []keys_h;
    delete []values_h;
    delete []check_h;


    return 0;
}




