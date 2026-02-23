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

Use a subset of cuda to implement host/device code run on gpu.

limitations:
- do not use cuda Graph
- use cuda streams for manageing multiple kernel launches
