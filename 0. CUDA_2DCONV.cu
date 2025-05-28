#include "cuda_runtime.h" 
#include "device_launch_parameters.h"
#include <omp.h> 
#include <cmath>
#include <stdio.h> 
#include <stdlib.h> 
#include <string.h>
#include <iostream>
#include <vector>
#include <chrono>

#define NUM_CPU_THREADS 4
#define IMG_ROW 1024
#define IMG_COL 1024
#define TILE 32

// gridDim = (4, 4)
// blockDim = (16, 16)
__global__ void WindowColvolution(float* _img, float* _window, float* _result,
                                  int img_ysize, int img_xsize, 
                                  int filter_dim) {

    __shared__ float shm[TILE][TILE];

    int filter_rad = filter_dim / 2;

    int Wp = TILE - 2 * filter_rad;
    int out_x = blockIdx.x * Wp + threadIdx.x - filter_rad;
    int out_y = blockIdx.y * Wp + threadIdx.y - filter_rad;
    float sum = 0.0f; //쓰레드마다 생성 되는 개별 변수 (모든원소의 연산이 병렬로 처리되서 다다르게 계산되는구조가 맞음)

    int sx = threadIdx.x;
    int sy = threadIdx.y;

    for (int dy = sy; dy < TILE; dy += blockDim.y) {
        for (int dx = sx; dx < TILE; dx += blockDim.x) {
            int gx = blockIdx.x * Wp + dx - filter_rad;
            int gy = blockIdx.y * Wp + dy - filter_rad;
            if ((gx >= 0 && gx < img_xsize) && (gy >= 0 && gy < img_ysize)) {
                shm[dy][dx] = _img[gy * img_xsize + gx];
            }
            else {
                shm[dy][dx] = 0.0f;
            }
        }
    }
    __syncthreads();

    // 커널이 3*3 기준이면 일단 그 뭐야.... ㅇ,.ㅇ;;;;
    // shared에서 0~30번까지만 넣어줘야 되고
    // sum 에서는 
    
    if ((sx >= filter_rad && sx < Wp + filter_rad) &&
        (sy >= filter_rad && sy < Wp + filter_rad)) {
        for (int i = -filter_rad; i <= filter_rad; i++) {
            for (int j = -filter_rad; j <= filter_rad; j++) {
                sum += shm[sy +j][sx +i] * _window[(j + filter_rad) * filter_dim + (i + filter_rad)];
            }
        }
        __syncthreads();

        if (out_x >= 0 && out_x < img_xsize && out_y >= 0 && out_y < img_ysize) {
            _result[out_y * img_xsize + out_x] = sum;
        }

    }
}

float cal_gaussian(float x, float y, float sigma) {
    float sigma2 = sigma * sigma;
    float square = (x * x + y * y);
    float norm = (square*(-1)) / (2 * sigma2);
    float weight = 1/(std::sqrt(2 * 3.14 * sigma)) ;

    return weight * std::exp(norm);
}

std::vector<float> make_gaussiankernel(int kernel_size, float sigma) {

    std::vector<float> result(kernel_size * kernel_size, 0.0f);
    float sum = 0.0f;
    float temp = 0.0f;

    float size = (float)kernel_size / 2;
    for (int i = 0; i < kernel_size; i++) {
        for (int j = 0; j < kernel_size; j++) {
            temp = cal_gaussian(i - size, j - size, sigma);
            result[j * kernel_size + i] = temp;
            sum += temp;
        }
    }

    for (auto& v : result) {
        v /= sum;
    }

    return result;
}

void conv2d_cpu(const std::vector<float>& img, std::vector<float>& out,
    int H, int W, const std::vector<float>& kernel, int kdim) {
    int kr = kdim / 2;
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            float sum = 0;
            for (int j = -kr; j <= kr; j++) {
                for (int i = -kr; i <= kr; i++) {
                    int yy = y + j, xx = x + i;
                    float v = (yy >= 0 && yy < H && xx >= 0 && xx < W) ? img[yy * W + xx] : 0;
                    sum += v * kernel[(j + kr) * kdim + (i + kr)];
                }
            }
            out[y * W + x] = sum;
        }
    }
}

int main()
{
    float* da, * dc;
    float* dkernel;
    float sigma = 1;


    int kernel_size = 0;
    std::cout << "kernel_size 를 입력하세요: ";
    std::cin >>  kernel_size ;

    std::cout << "입력받은 값: " << kernel_size << "\n";

    std::vector<float> kernel(kernel_size* kernel_size);
    std::vector<float> A(IMG_COL *IMG_ROW);
    std::vector<float> C(IMG_COL * IMG_ROW, 0.0f);
    std::vector<float> C_cpu(IMG_COL * IMG_ROW, 0.0f);

    for (int i = 0; i < IMG_COL * IMG_ROW; ++i) {
        A[i] = (float)(rand() % 100);
    }

    kernel = make_gaussiankernel(kernel_size,sigma);

    // add matrix padding
    cudaMalloc(&da, IMG_ROW* IMG_COL * sizeof(float));
    cudaMalloc(&dc, IMG_ROW * IMG_COL * sizeof(float));
    cudaMalloc(&dkernel, kernel_size* kernel_size * sizeof(float));


    cudaMemcpy(da, A.data(), A.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dkernel, kernel.data(), kernel.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dc, C.data(), C.size() * sizeof(float), cudaMemcpyHostToDevice);


    int WP = TILE - 2*(kernel_size / 2);
    int blk_row_dim = (IMG_ROW + WP - 1) / WP;
    int blk_col_dim = (IMG_COL + WP - 1) / WP;

    dim3 gridDim(blk_col_dim, blk_row_dim);
    dim3 blockDim(TILE, TILE);

    // GPU 시간 측정용 이벤트 생성
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // CPU 시간 측정 시작
    auto cpu_start = std::chrono::high_resolution_clock::now();

    // CPU conv
    conv2d_cpu(A, C_cpu, IMG_ROW, IMG_COL, kernel, kernel_size);

    auto cpu_end = std::chrono::high_resolution_clock::now();

    // GPU conv 시작 이벤트 기록
    cudaEventRecord(start);

    WindowColvolution << <gridDim, blockDim >> > (da, dkernel, dc, IMG_ROW, IMG_COL, kernel_size);

    // GPU conv 끝 이벤트 기록
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    // CPU 시간 측정 끝
    

    // 결과 복사
    cudaMemcpy(C.data(), dc, C.size() * sizeof(float), cudaMemcpyDeviceToHost);

    // GPU 경과 시간(ms)
    float gpu_ms = 0;
    cudaEventElapsedTime(&gpu_ms, start, stop);

    // CPU 경과 시간(ms)
    double cpu_ms = std::chrono::duration<double, std::milli>(cpu_end - cpu_start).count();

    // max error 계산 (기존)
    float max_err = 0;
    for (int i = 0; i < IMG_ROW * IMG_COL; i++)
        max_err = fmaxf(max_err, fabsf(C_cpu[i] - C[i]));

    // 출력
    printf("max error = %f\n", max_err);
    printf("CPU time  = %.3f ms\n", cpu_ms);
    printf("GPU time  = %.3f ms\n", gpu_ms);

    // 클린업
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(da);
    cudaFree(dc);
    cudaFree(dkernel);

    return 0;


}

