## File structures

cmake/
includes/
src/
    sub0/
        inc/
        src/
        test/
        CMakeLists.txt
    sub1/
        inc/
        src/
        test/
        CMakeLists.txt
    CMakeLists.txt
CMakeLists.txt
thirdparty


## Code style

C/C++:
- use camelCase for functions and methods, name starts with lower case, e.g. `myFunc`
- use CamelCase for class/structs, e.g `MyClass`
- use snake case for variables/arguments, e.g. `my_var`
- always occupy an entire line for `{}` for class/namespace/struct/function/ifelse/for/switch blocks.
    e.g.
    ```cpp
    void func()
    {

    }
    ```


## Development
- follow the file structure convention, src/demo is an example.
- for host/device code, use ".h" file for header files, use ".cu" file for source files, use ".cpp" file for test files and write tests with gtest.
- use `cuda_library` and `cuda_gtest` to define library and tests in cmake, examples can be found in src/demo/CMakeLists.txt
- use cmake with ninja to build the project
```
cmake --preset default
cd build
ninja
ctest
```
- write kernel and host code using cuda. write test using cpp with gtest

## cuda-like Programming model

We use a custom hardware similar to nvidia's GPU, use a subset of cuda to implement host/device code run on gpu and always consider the following limitations and the hardware abstraction.
- in general, use CUDA C++ API.
- do not optimze for speed unless you are told explicitly, readibility, simplicity and testibility is the rules of thumb.
- use 
- Use C++17 standard features compatible with CUDA C++, avoid using lambda functions.

As a supplementary, we added some keywords in src/common/inc/predefine to improve readibility. Always use them if possible.
- `__mem0__` to `__mem4__` for local memory in each thread. always consider reading/writing local memory is extremely fast.
- add `__DDR` to a memory if it is in DDR, other it's regarded as a local memory.
    e.g. 
    ```
    __global__ void kernel(const __DDR int *a, __DDR int *b);
    ```


### limitations:
- do not use cuda Graph
- use cuda streams for managing multiple kernel launches aynchronously

### Hardware Abstraction

#### Hardware Architecture
- StreamProcessor: the device has 32 StreamProcessors in total, four StreamProcessor forms a Processor Cluster, each StreamProcessor has 4 identical cores where cores in the same processor share the same 24M data memory.
- Global memory: The global device memory is DDR, it has 16G bytes in total, 64 bit address based,  
- Shared memory & local memory: the 24M data memory inside each StreamProcessor is 32-bit addressed, has 24 banks, and is logically divided to shared memory and local memory. To avoid bank conflicts，we logically split them to 4 parts:
    a. core0: bank0~5; core1: bank6~11; core2: bank12~17；core3: bank18~23
    b. threads run in different cores use different banks.
    c. for each core, 5 banks out of 6 is regarded as the local memory, the remaning bank is the shared memory, one can use `__mem0__`/`__mem1__`/`__mem2__`/`__mem3__`/`__mem4__`/ to assign bank id to a local memory, use `__shared__` to allocate a shared memory variable.
    for example,
    ```cuda
    __global__ void kernel(int *a) 
    {
        __mem0__ int local_a[100];
        __shared__ int shared_var;
    }
    ```
    The total size of local memory arrays per core(each has at maximum 32 threads) (across __mem0__ to __mem4__) strictly must not exceed  256KB. Avoid allocating overly large arrays in local memory.

    d. different with cuda, reading/writing local memory is extremely fast. 

#### Thread Hierarchy

The thread hierarchy is as follows: Thread -> Block -> Cluster -> Grid.
- Thread: The smallest execution unit for parallel computation. Each thread independently executes the same kernel function code.
    - has a private register file
    - has local memory
    - at any moment, one core can only run one thread. 
    - threads in each core are executed concurrently via time-sharing scheduling
    - each core can support up to 32 concurrent threads.

- Block: A unit composed of a group of cooperating threads.
    - has shared memory
    - supports synchronization primitives(`__syncthreads()`)
    - threads within a block have three-dimensional indices(`threadIdx.x/y/z`)
    - all threads within a block must be issued together into a single StreamProcessor, therefore, a block can have at most 128 threads
    - a StreamProcessor can execute up to 32 blocks concurrently, each block has a separate shared memory.
    - these 32 blocks may belong to at most 16 different kernels

- Cluster: A physical execution group composed of multiple thread blocks, working collaboratively within the same processing cluster

    - has distributes shared memory
    - supports atomic operations across blocks for inter-block synchronization
    - blocks within a cluster are issued sequentially into the Stream Cluster.
    - a Cluster supports a maximum of 32 blocks
    - each StreamProcessor is assigned only one block
    - haredware constraints limit clusterDim.x to values of 2, 4, 6, 8, 16, or 32; and clusterDim.y and clusterDim.z can only be 1.
    - use `cudaLaunchKernelEx` to launch cuda kernel with a thread block cluster configuration.

- Grid: A logical execution unit composed of multiple thread blocks, representing the entire scope of kernel function execution.
    - has access to global memory
    - uses three-dimensional block indices(`blockIdx.x/y/z`)

#### Best practice

1. always use `memcpy_async` with barrier for copy data between global memory and local memory
    ```
    __global__ void vadd_kernel(const __DDR int *a)
    {
        using barrier_t = cuda::barrier<cuda::thread_scope_block>;
        __mem0__ int local_a[256];

        auto block = cooperative_groups::this_thread_block();
        __shared__ barrier_t bar;
        if (block.thread_rank() == 0)
        {
            init(&bar, block.size());
        }
        block.sync();
        cuda::memcpy_async((int *)local_a, a, 256 * sizeof(int), bar);
        bar.arrive_and_wait();
    }

    ```
2. always use add explicit bank id to local memory, though in cuda we just ignore it. e.g. `__mem0__ int local_a[100];`, size of local memory must be constant.
3. do not use any cuda intrinsics
