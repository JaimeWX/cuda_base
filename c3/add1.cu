#include <math.h>
#include <stdio.h>

const double EPSILON = 1.0e-15;
const double a = 1.23;
const double b = 2.34;
const double c = 3.57;

__global__ void add(const double *x, const double *y, double *z);
void check(const double *z, const int N);

int main(void)
{
    const int N = 100000000;
    const int M = sizeof(double) * N;
    double *h_x = (double*) malloc(M);
    double *h_y = (double*) malloc(M);
    double *h_z = (double*) malloc(M);

    for (int n = 0; n < N; ++n)
    {
        h_x[n] = a;
        h_y[n] = b;
    }

    double *d_x, *d_y, *d_z;
    /*
        第一个参数address：
            待分配设备内存的指针。因为内存（地址）本身就是一个指针，所以待分配设备内存的指针就是指针的指针，即双重指针
        第二个参数size：
            待分配内存的字节数
        返回值是一个错误代号
            如果调用成功，返回cudaSuccess，否则返回一个代表某种错误的代号
    */
    cudaMalloc((void **)&d_x, M);// 分配显存
    cudaMalloc((void **)&d_y, M);
    cudaMalloc((void **)&d_z, M);
    /*
        第一个参数dst是目标地址
        第二个参数src是源地址
        第三个参数count是复制数据的字节数
        第四个参数kind一个枚举类型的变量，标志数据传递方向
            cudaMemcpyDefault
        返回值是一个错误代号。如果调用成功，返回cudaSuccess
    */
    cudaMemcpy(d_x, h_x, M, cudaMemcpyHostToDevice); // 该函数的作用是将一定字节数的数据从源地址所指缓冲区复制到目标地址所指缓冲区
    cudaMemcpy(d_y, h_y, M, cudaMemcpyHostToDevice);

    const int block_size = 128; // 具有128个线程的一维线程块
    const int grid_size = N / block_size; // 总共有 N / block_size 个具有128个线程的一维线程块
    add<<<grid_size, block_size>>>(d_x, d_y, d_z);

    cudaMemcpy(h_z, d_z, M, cudaMemcpyDeviceToHost);
    check(h_z, N);

    free(h_x);
    free(h_y);
    free(h_z);

    // 参数address就是待释放的设备内存变量（不是双重指针）。
    // 返回值是一个错误代号。如果调用成功，返回cudaSuccess
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_z);
    return 0;

}

/*
    “单指令-多线程”
        与add.cpp中的add()相比，去掉了一层循环
        只需要将数组元素与线程指标一一对应即可
*/
__global__ void add(const double *x, const double *y, double *z)
{   
    /*
        赋值号右边只出现标记线程的内建变量，左边的n是后面代码中将要用到的数组元素指标
            在这种情况下，
                第0号线程块中的blockDim.x个线程对应于第0个到第blockDim.x-1个数组元素，
                第1号线程块中的blockDim.x个线程对应于第blockDim.x个到第2*blockDim.x-1个数组元素，
                第2号线程块中的blockDim.x个线程对应于第2*blockDim.x个到第3*blockDim.x-1个数组元素，依此类推。
                
        这里的blockDim.x等于执行配置中指定的（一维）线程块大小。核函数中定义的线程数目与数组元素数目一样，都是10^8。
            在将线程指标与数据指标一一对应之后，就可以对数组元素进行操作了
    */
    const int n = blockDim.x * blockIdx.x + threadIdx.x;
    z[n] = x[n] + y[n];
}

void check(const double *z, const int N)
{
    bool has_error = false;
    for (int n = 0; n < N; ++n)
    {
        if (fabs(z[n] - c) > EPSILON)
        {
            has_error = true;
        }
    }
    printf("%s\n", has_error ? "Has errors" : "No errors");
}