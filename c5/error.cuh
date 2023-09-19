#pragma once // 一个预处理指令，其作用是确保当前文件在一个编译单元中不被重复包含
#include <stdio.h>

/*
    该宏函数的名称是CHECK，参数call是一个CUDA运行时API函数。
        定义了一个cudaError_t类型的变量error_code，并初始化为函数call的返回值
        判断该变量的值是否为cudaSuccess。如果不是，报道相关文件、行数、错误代号及错误的文字描述并退出程序。
        cudaGetErrorString显然也是一个CUDA运行时API函数，作用是将错误代号转化为错误的文字描述。
*/
#define CHECK(call)                                   \
do                                                    \
{                                                     \
    const cudaError_t error_code = call;              \
    if (error_code != cudaSuccess)                    \
    {                                                 \
        printf("CUDA Error:\n");                      \
        printf("    File:       %s\n", __FILE__);     \
        printf("    Line:       %d\n", __LINE__);     \
        printf("    Error code: %d\n", error_code);   \
        printf("    Error text: %s\n",                \
            cudaGetErrorString(error_code));          \
        exit(1);                                      \
    }                                                 \
} while (0)
