#include <stdio.h>

__global__ void hello_from_gpu()
{
    const int b = blockIdx.x;
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    printf("hello world from block-%d and thread-(%d,%d)\n", b, tx, ty);
}

int main(int argc, char const *argv[])
{   
    const dim3 block_size(2,4);

    hello_from_gpu<<<1, block_size>>>(); 

    // 一个cuda的运行时API函数，作用是同步主机与设备（促使缓冲区刷新）
    cudaDeviceSynchronize();
    return 0;
}