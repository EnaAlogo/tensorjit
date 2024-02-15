# tensorjit(or whatever)
# The project was made for educational purposes. The goal was to make a general easy to use tool that jits tensor operations using NVRTC and [jitify](https://github.com/NVIDIA/jitify)
# Short-term Goals
 * Refactor for more flexibility and extensibility 
 * Python and possibly R bindings
 * Creating a higher level abstraction and making the API as user friendly as possible (e.g. passing parameters by name using a map)
 * Ability to extract the generated PTX assembly for debugging or even educational purposes, the challenge is that different kernels are being compiled and cached for different inputs for the same operation.
 * Add a "helper functions" arguement for the element-wise operation similar to the one used for reductions
# Long-term Goals :
 * Support for AMD gpus
 * More kernel options like:
   - a kernel that can use rng (curand)
   - indexing & scatter/gather kernels (e.g. scatter max add etc)
 * Interoperability with most major ML libraries/frameworks (e.g. pytorch , tensorflow)
 * Supporting just in time compilation /w vectorization for cpus using google/highway or std::simd(if that ever happens)


# Example Usage
```cpp
   auto alloc = [](size_t b) {
        void* block;
        MEGU_CUDA_CHECK(cudaMalloc(&block, b));
        return block;
    };
    auto dealloc = [](void* block) {
        MEGU_CUDA_CHECK(cudaFree(block));
    };
    /*
    * create a custom allocator for the library to use if it needs to allocate memory
    * will probably make this more object oriented to be easily extensible from python
    */
    auto allocator = InjectedAllocator(alloc, dealloc);

    //this is just a shared_ptr for now
    mem_ptr data = allocator(25 * sizeof(float));

    /*
    * create a tensor that can either own the memory or just borrow it from a raw pointer (in *this case we move it)
    *this will obviously change since shared pointers are very restrictive and we want to be *able to return the allocated memory to the consumer to manage it however they want thats *the entire reason of the user provided allocator
    */
    JitTensor x( std::move(data), {25},1, 0, Float); 

    //define your function with your mutable and (default)constant parameters
    //the types will be infered 
    cuda::jit::JitFunction fn(/*function name=*/"fill_kernel", 
        /*function args=*/"mut x , val", 
        /*function code=*/"x = val;"); 

    Scalar param = -23.f;

    //create your array of arguements that can be either tensors or scalars
    cuda::jit::JitArg ops[] = { std::move(x) , param };
     
    //start building the operation
    cuda::jit::JitDeez builder(ops, fn.function_args(), allocator);
    //add any extra options you want eg the broadcast shape (if you dont they will be infered)
    builder.provideDtype(Float); 
    
    //prepare the single arguement that the jit function expects
    ElementWiseArgument arg; 
    cuda::jit::JitFunctionArg injit = builder.setArgs(arg);  

    //call the function with your own stream and an option for the function to produce a vectorized
    //kernel when possible (eg if all tensors are contiguous) usually prefared for smaller operations
    fn(injit, /*stream=*/ nullptr, /*try_vectorize=*/true);

    //it can perform broadcasting (see numpy) and also allocate the output
    //of the correct shape if you pass an empty `JitArg` using the provided allocator. 
    //If the output was generated inside the function you can extract it by specifying the index of the argument
    JitTensor output = builder.realizeOutput(0); 
```
