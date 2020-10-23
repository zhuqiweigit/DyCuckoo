#ifndef __GPU_TIMER_H__
#define __GPU_TIMER_H__

#include <string>
#include <iostream>

class GpuTimer {
	cudaEvent_t start;
	cudaEvent_t stop;

public:

	GpuTimer() {
		cudaEventCreate(&start);
		cudaEventCreate(&stop);
	}

	~GpuTimer() {
		cudaEventDestroy(start);
		cudaEventDestroy(stop);
	}

	void Start() {
		cudaEventRecord(start, 0);
	}

	void Stop() {
		cudaEventRecord(stop, 0);
	}

	float Elapsed() {
		float elapsed;
		cudaEventSynchronize(stop);
		cudaEventElapsedTime(&elapsed, start, stop);
		return elapsed / 1000;
	}

};

#endif  /* __GPU_TIMER_H__ */
