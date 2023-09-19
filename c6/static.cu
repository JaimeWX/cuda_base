#include "error.cuh"
#include <stdio.h>
// 其中，修饰符__device__说明该变量是设备中的变量，而不是主机中的变量
__device__ int d_x = 1;
__device__ int d_y[2];

void __global__ my_kernel(void)
{
    d_y[0] += d_x;
    d_y[1] += d_x;
    printf("d_x = %d, d_y[0] = %d, d_y[1] = %d.\n", d_x, d_y[0], d_y[1]);
}

int main(void)
{
    int h_y[2] = {10, 20};
    /*
        cudaError_t cudaMemcpyToSymbol
        (
            const void* symbol, //静态全局内存变量名或常量内存变量的变量名
            const void* src, //主机内存缓冲区指针
            size_t count, //复制的字节数
            size_t offset = 0, //从symbol对应设备地址开始偏移的字节数
            cudaMemcpyKind kind = cudaMemcpyHostToDevice //可选参数
        )
    */
    CHECK(cudaMemcpyToSymbol(d_y, h_y, sizeof(int) * 2));
    
    my_kernel<<<1, 1>>>();
    CHECK(cudaDeviceSynchronize());
    
    /*
        cudaError_t cudaMemcpyFromSymbol
        (
            void* dst, //主机内存缓冲区指针
            const void* symbol, //静态全局内存变量名或常量内存变量的变量名
            size_t count, //复制的字节数
            size_t offset = 0, //从symbol对应设备地址开始偏移的字节数
            cudaMemcpyKind kind = cudaMemcpyHostToDevice //可选参数
        )
    */

    CHECK(cudaMemcpyFromSymbol(h_y, d_y, sizeof(int) * 2));
    printf("h_y[0] = %d, h_y[1] = %d.\n", h_y[0], h_y[1]);
    
    return 0;
}

