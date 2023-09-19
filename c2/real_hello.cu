#include <stdio.h>

__global__ void hello_from_gpu()
{
    printf("hello world from the gpu\n");
}

int main(int argc, char const *argv[])
{   
    // 三括号中的第一个数字可以看作线程块的个数，第二个数字可以看作每个线程块中的线程数
    hello_from_gpu<<<1, 1>>>(); 

    // 一个cuda的运行时API函数，作用是同步主机与设备（促使缓冲区刷新）
    cudaDeviceSynchronize();
    return 0;
}
