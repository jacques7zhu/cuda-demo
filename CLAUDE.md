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

## cuda-like Programming model

We use a custom haredware similar to nvidia's GPU, use a subset of cuda to implement host/device code run on gpu and always consider the following limitations and the hardware abstration.

### limitations:
- do not use cuda Graph
- use cuda streams for managing multiple kernel launches

### Hardware Abstraction

#### Hardware Architechure
- StreamProcessor: the device has 32 StreamProcessors in total, four StreamProcessor forms a Processor Cluster, each StreamProcessor has 4 identical cores where cores in the same processor share the same 24M data memory.
- Global memory: The global device memory is DDR, it has 16G bytes in total, 64 bit address based.
- Shared memory & local memory: the 24M data memory inside each StreamProcessor is 32-bit addressed, has 24 banks, and is logically divided to shared memory and local memory. To avoid bank conflicts，we logically split them to 4 parts:
    a. core0: bank0~5; core1: bank6~11; core2: bank12~17；core3: bank18~23
    b. threads run in different cores use different banks.
    c. for each core, 5 banks out of 6 is regarded as the local memory, the remaning bank is the shared memory, one can use `__mem0~4__` to assign bank id to a local memory, use `__shared__` to allocate a shared memory variable.
    for example,
    ```cuda
    __global__ void kernel(int *a) 
    {
        __mem0__ int local_a[100];
        __shared__ int shared_var;
    }
    ```
    d. different with cuda, reading/writing local memory is extremly fast. 

#### Thread Hierarchy

The thread hierarchy is as follows: Thread -> Block -> Cluster -> Grid.
- Thread: The smallest execution unit for paralle computation. Each thread independently executes the same kernel function code.
    - has a private register file
    - has local memory
    - at any moment, one core can only run one thread. 
    - threads in each core are executed concurrently via time-sharing scheduling
    - each core can support up to 32 concurent threads.

- Block: A unit composed of a group of cooperating threads.
    - has shared memory
    - supports synchornization primitives(`__syncthreads()`)
    - threads within a block have three-dimensinal indices(`threadIdx.x/y/z`)
    - all threads within a block must be issued together into a single StreamProcessor, therefore, a block can have at most 128 threads
    - a StreamProcessor can execute up to 32 blocks concurrently
    - these 32 blocks may belong to at most 16 different kernels

- Cluster: A physical execution group composed of multiple thread blocks, working collaboratively within the same processing cluster
    - has distributes shared memory
    - supports atomic operations across blocks for inter-block synchronization
    - blocks within a cluster are issued sequentially into the Stream Cluster.
    - a Cluster supports a maximum of 32 blocks
    - each StreamProcessor is assigned only one block
    - haredware constraints limit clusterDim.x to values of 2, 4, 6, 8, 16, or 32; and clusterDim.y and clusterDim.z can only be 1.

- Grid: A logical exection unit composed of multiple thread blocks or clusters, representing the entire scope of kernel function exection.
    - has access to global memory
    - uses three-dimensional block indices(`blockIdx.x/y/z`)

#### Best practice

1. always use `memcpy_async` with barrier for copy data between global memory and local memory
2. always use add explicit bank id to local memory, though in cuda we just ignore it. e.g. `__mem0__ int local_a[100];`
3. do not use any cuda intrinsics
