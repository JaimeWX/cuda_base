cudaEvent_t start, stop; # 定义了两个CUDA事件类型（cudaEvent_t）的变量start和stop
CHECK(cudaEventCreate(&start)); # 用cudaEventCreate函数初始化start
CHECK(cudaEventCreate(&stop)); # 用cudaEventCreate函数初始化stop
CHECK(cudaEventRecord(start)); # 将start传入cudaEventRecord函数，在需要计时的代码块之前记录一个代表开始的事件
cudaEventQuery(start); // cannot use the macro function CHECK here

// The code block to be timed

CHECK(cudaEventRecord(stop)); # 将stop传入cudaEventRecord函数，在需要计时的代码块之后记录一个代表结束的事件
CHECK(cudaEventSynchronize(stop)); # cudaEventSynchronize函数让主机等待事件stop被记录完毕

# 调用cudaEventElapsedTime函数计算start和stop这两个事件之间的时间差（单位是ms）并输出到屏幕
float elapsed_time;
CHECK(cudaEventElapsedTime(&elapsed_time, start, stop));
printf("Time = %g ms.\n", elapsed_time);

# 调用cudaEventElapsedTime函数计算start和stop这两个事件之间的时间差（单位是ms）并输出到屏幕
CHECK(cudaEventDestroy(start));
CHECK(cudaEventDestroy(stop));