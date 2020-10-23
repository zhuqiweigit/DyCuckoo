#include <helper_functions.h>
#include <helper_cuda.h>
#include <stdint.h>
#include "../tools/gputimer.h"
#include "../data/data_layout.cuh"
#include "../core/dynamic_cuckoo.cuh"
namespace ch = cuckoo_helpers;
using namespace std;
class DynamicTest {
public:
    using key_t = DataLayout<>::key_t;
    using value_t = DataLayout<>::value_t;
    static constexpr uint32_t val_lens = DataLayout<>::val_lens;

    int r = 2;
    int batch_size = 100000;  //smaller batch size: 2e4 4e4 6e4 8e4 10e4
    double lower_bound = 0.5;  //lower bound: 0.3 0.35 0.4 0.45 0.5
    double upper_bound = 0.85; //upper bound: 0.7 0.75 0.8 0.85 0.9
    int pool_len = 0;
    key_t *keys_pool_d;
    value_t *value_pool_d, *check_pool_d;
    double init_fill_factor = 0.85;
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

    void batch_check(value_t *check_pool_d, int32_t single_batch_size, uint32_t offset) {
        uint32_t error_cnt = 0;
        value_t *check_pool_h = new value_t[single_batch_size];
        cudaMemcpy(check_pool_h, check_pool_d + offset, sizeof(value_t) * single_batch_size, cudaMemcpyDeviceToHost);
        for (int i = 0; i < single_batch_size; i++) {
            for(int j = 0; j < val_lens; j++){
                if(check_pool_h[i].data[j] != i + 5 + offset){
                    ++error_cnt;
                    break;
                }
            }
        }
        if (error_cnt != 0) {
            printf("num error:%d \n", error_cnt);
        } else {
            printf("batch check ok\n");
        }
        delete[] check_pool_h;
    }

    void batch_test() {
        DynamicCuckoo<512, 512> dy_cuckoo((uint32_t)batch_size * 10 / init_fill_factor, batch_size, lower_bound, upper_bound);
        int32_t batch_num = pool_len / batch_size;
        int32_t batch_round = batch_num / 10;
        GpuTimer timer;
        timer.Start();
        for (int repeat = 0; repeat < 5; repeat++) {
            for (int32_t batch_round_ptr = 0; batch_round_ptr < batch_round; ++batch_round_ptr) {
                int batch_ptr = batch_round_ptr * 10;
                for (int j = 0; j < 10; j++) {
                    dy_cuckoo.batch_insert(keys_pool_d + (batch_ptr + j) * batch_size,
                                           value_pool_d + (batch_ptr + j) * batch_size, batch_size);
                }
                for (int j = 0; j < 10; j++) {
                    dy_cuckoo.batch_search(keys_pool_d + (batch_ptr + j) * batch_size,
                                           check_pool_d + (batch_ptr + j) * batch_size, batch_size);
                }
                for (int j = 0; j < r; j++) {
                    dy_cuckoo.batch_delete(keys_pool_d + (batch_ptr + j) * batch_size, nullptr, batch_size);
                }
                //batch_check(check_pool_d,  10 * batch_size, batch_ptr * batch_size);
            }
            //cudaMemset(check_pool_d, 0, sizeof(value_t) * pool_len);
            for (int32_t batch_round_ptr = 0; batch_round_ptr < batch_round; ++batch_round_ptr) {
                int batch_ptr = batch_round_ptr * 10;
                for (int j = 0; j < r; j++) {
                    dy_cuckoo.batch_insert(keys_pool_d + (batch_ptr + j) * batch_size,
                                           value_pool_d + (batch_ptr + j) * batch_size, batch_size);
                }
                for (int j = 0; j < 10; j++) {
                    dy_cuckoo.batch_search(keys_pool_d + (batch_ptr + j) * batch_size,
                                           check_pool_d + (batch_ptr + j) * batch_size, batch_size);
                }
                for (int j = 0; j < 10; j++) {
                    dy_cuckoo.batch_delete(keys_pool_d + (batch_ptr + j) * batch_size, nullptr, batch_size);
                }
                //batch_check(check_pool_d, 10 * batch_size, batch_ptr * batch_size);
            }
        }

        timer.Stop();
        double diff = timer.Elapsed() * 1000000;
        printf("<throughtput> %.2lf %.2lf\n", (double) diff,
               (double) (batch_round * 10 * 4 * batch_size + batch_round * r * 2 * batch_size) * 5 / diff);
    }
};




int main(int argc, char** argv) {
    using test_t = DynamicTest;

    if (argc < 7)
    {
        cout << "para error\n" << endl;
        return -1;
    }

    test_t dy_test;
    char* file_name = argv[1];
    int pool_len = atoi(argv[2]);
    dy_test.pool_len = pool_len;
    dy_test.r = atoi(argv[3]);
    //batch size: 2e5 4e5 6e5 8e5 10e5
    dy_test.batch_size = atoi(argv[4]) / 10;
    dy_test.lower_bound = atof(argv[5]);
    dy_test.upper_bound = atof(argv[6]);
    dy_test.init_fill_factor = atof(argv[7]);
    test_t::key_t* keys_h = test_t::read_data(file_name, pool_len);
    test_t::value_t *values_h = new test_t::value_t [pool_len], *check_h = new test_t::value_t [pool_len];
    for(int i = 0; i < pool_len; i++){
        for(int j = 0; j < DataLayout<>::val_lens; j++){
            values_h[i].data[j] = i + 5;
            check_h[i].data[j] = 0;
        }
    }
    cudaMalloc((void**)&(dy_test.keys_pool_d), sizeof(test_t::key_t) * pool_len);
    cudaMalloc((void**)&(dy_test.value_pool_d), sizeof(test_t::value_t) * pool_len);
    cudaMalloc((void**)&(dy_test.check_pool_d), sizeof(test_t::value_t) * pool_len);
    cudaMemcpy(dy_test.keys_pool_d, keys_h, sizeof(test_t::key_t) * pool_len, cudaMemcpyHostToDevice);
    cudaMemcpy(dy_test.value_pool_d, values_h, sizeof(test_t::value_t) * pool_len, cudaMemcpyHostToDevice);
    dy_test.batch_test();

    delete []keys_h;
    delete []values_h;
    delete []check_h;
    return 0;
}




