#include "../include/jit_builder.hpp"
#include "../common/shape_tricks.hpp"
#include "../common/string_util.hpp"
#include "../include/tensor.h"
#include <ranges>
#include <cuda_runtime.h>


namespace megu::cuda::jit
{

    [[nodiscard]] JitTensor JitArg::realize() {
        MEGU_ENSURE(!is_scalar() && tensor_.realized(), "[JIT] You are trying to realize a scalar or undefined tensor");
        return std::move(tensor_);
    }

    void JitArg::allocate(InjectedAllocator const& alloc, LongArrayView shape, Device dev, dtype_t dt)
    {
        assert(t_ == jitType_t::TENSOR);
        
        int64_t size = shape_tricks::prod(shape);
        size_t tsize = megu::detail::type_size(dt);

        mem_ptr data = alloc(tsize * size);

        tensor_ = JitTensor(std::move(data), shape, memory::computeContiguousStrides(shape), dev, dt);

        t_ = jitType_t::TENSOR;
    }

    void JitArg::maybe_broadcast_to(LongArrayView shape)
    {
        if (!is_scalar() && shape != this->shape())
        {
            tensor_ = tensor_.broadcast_to(shape); 
            t_ = jitType_t::TENSOR;
        }
    }
    int64_t JitDeez::numel()const {
        return shape_tricks::prod(*shape_);
    }

    bool JitDeez::isSquashableDimension(size_t i, const ShapeVector& shape)const 
    {
        for (int ii = 0; ii < operands_.size(); ++ii) {
            if (isScalar(ii) || isPointer(ii))
                continue;
            if (operands_[ii].stride(i) * shape[i] != operands_[ii].stride(i - 1))
                return false;
        }
        return true;
    }


    JitDeez& JitDeez::checkDevices()
    {
        MEGU_ENSURE(std::adjacent_find(operands_.begin(), operands_.end(),
            [](const JitArg& l, const JitArg& r) -> bool
            {
                if (l.is_scalar() || r.is_scalar() || !l.exists() || !r.exists())
                    return false;
                return l.device() != r.device();
            }
        ) == operands_.end(),
                "[JIT] Found tensors that belong on different devices");
        return *this;
    }

    JitDeez& JitDeez::checkSameDtypes()
    {
        MEGU_ENSURE(std::adjacent_find(operands_.begin(), operands_.end(),
            [](const JitArg& l, const JitArg& r) -> bool
            {
                return l.dtype() != r.dtype();
            }
        ) == operands_.end(),
                "All arguements must have the same dtype");
        return *this;
    }

    
    dtype_t JitDeez::findCommonDtype() const {
        std::optional<dtype_t> r{};
        for (int i = 0; i < operands_.size(); ++i) {
            if (isOutput(i) || isScalar(i))
                continue;
            const auto& ref = operands_[i];
            r = r ? megu::detail::promote_type(r.value(), ref.dtype()) : ref.dtype();
        }
        if (!r) {//go pick one from scalar or output args whatever
            for (int i = 0; i < operands_.size(); ++i) {
                if (isOutput(i) && operands_[i].exists()) {
                    const auto& ref = operands_[i]; 
                    r = r ? megu::detail::promote_type(r.value(), ref.dtype()) : ref.dtype(); 
                }
                else if (isScalar(i)) {
                    const auto& ref = operands_[i]; 
                    r = r ? megu::detail::promote_type(r.value(), ref.dtype()) : ref.dtype(); 
                }
            }
        }
        MEGU_ENSURE(r, "Unexpected empty optional when trying to find common dtype");
        return r.value();
    }


    JitDeez& JitDeez::checkShapes()
    {
        for (int i = 0; i < operands_.size(); ++i) {
            //inputs and raw argumenets dont have to get shape check(?)
            if (!isOutput(i) || isPointer(i))
                continue;
            MEGU_ENSURE(operands_[i].shape() == LongArrayView(shape_.value()),
                "Shape check failed , all output parameters must have the same shape : ",
                shape_.value(), " but found an output of shape : ", operands_[i].shape());
        }
        return *this;
    }

    void JitDeez::broadcastAndAllocate()
    {

        checkDevices();

        if (!compute_type_)
            compute_type_.emplace(findCommonDtype());
        if (!shape_)
            shape_.emplace(getBroadcastShape());
        if (!device_)
            device_.emplace(findDevice());

        bool should_braodcast = false;
        for (int i = 0; i < operands_.size(); ++i)
        {
            if (isInput(i))
                continue;
            if (isPointer(i))
                continue;

            should_braodcast = true;
        }

        dtype_t compute_type = compute_type_.value();
        LongArrayView shape = shape_.value();
        Device device = device_.value();

        for (int i = 0; i < operands_.size(); ++i)
        {
            if (isScalar(i))
                continue;
            if (should_braodcast && isInput(i) && !isPointer(i))
            {

                operands_[i].maybe_broadcast_to(shape);
                continue;
            }

            if (operands_[i].exists() && !is_raw[i])
            {
                MEGU_ENSURE(operands_[i].shape() == shape,
                    "Shape check failed , all output parameters must have the same shape : ",
                    shape_.value(), " but found an output of shape : ", operands_[i].shape());
                continue;
            }
            if (!operands_[i].exists()) {
                operands_[i].allocate(allocator_ , shape, device, compute_type);
            }
        }
    }
    Device JitDeez::findDevice() const {
        for (int i = 0; i < operands_.size(); ++i)
        {
            if (!operands_[i].is_scalar() && operands_[i].exists())
            {
                return operands_[i].device();
            }
        }
        int dev;
        MEGU_CUDA_CHECK(cudaGetDevice(&dev));
        return dev ;
    }


    constexpr static bool is_pointer(std::string_view range)
    {
        return range.find('*') != std::string_view::npos;
    }
    constexpr static bool is_input(std::string_view  name)
    {
        std::size_t const pos = name.find("mut");  
        return pos == std::string::npos || pos == name.size() - 3 ? true 
            : !std::isspace(name[pos + 3]) && name[pos + 3] != '*';
    }
     
    JitFunctionArg JitDeez::fromEltwiseArg(  
        ElementWiseArgument const& arg, 
        ArrayRef<std::string> args,
        Device dev,
        std::optional<std::bitset<24>> scalar_indices,
        std::optional<std::vector<Scalar>> scalar_args ) {
        int noutputs = 0; 
        JitFunctionArg out( arg, dev );
        
        int i = 0;

        MEGU_ENSURE(scalar_indices.has_value() == scalar_indices.has_value(),
            "Scalar args and indices have to be both undefined or both defined");

        if (scalar_args) {
            out.scalars = *scalar_args;
            out.is_scalar = *scalar_indices;
        }
        for (std::string const& v : args) {
            if (out.is_scalar[i]) {
                out.is_const[i] = true;
                continue;
            }
            out.is_const[i] = is_input(v);
            noutputs += int(!is_const[i]);
           
            if (is_const[i]) {
                MEGU_ENSURE(i < arg.nargs(), "[JIT]Argument mismatch, parsed arg",
                    i, "  but only ", arg.nargs(), " were given.");
            }
            MEGU_ENSURE(arg.data[i] != nullptr, "[JIT] All data pointers in eltwise arg have to be non null");
            ++i;
        }
        MEGU_ENSURE(i + out.is_scalar.count() == operands_.size(), "[JIT]Argument mismatch, parsed ",
            i + out.is_scalar.count(), " arguements  but ", operands_.size(), " were given.");

        MEGU_ENSURE(noutputs != 0, "[JIT] Operation with no output parameters is not allowed (and doesnt even make sense duh)");

        return out;
    }

    JitDeez::JitDeez(JitArrayView ops , ArrayRef<std::string> args, InjectedAllocator cator)
    : operands_{ops}, allocator_(std::move(cator))
    {
        int noutputs = 0;
        int i = 0;
        for (std::string const& v : args) { 
            is_const[i] = is_input(v); 
            is_raw[i] = is_pointer(v); 
            noutputs += int(!is_const[i]); 
            if (!is_const[i] || is_raw[i]) 
            {
                MEGU_ENSURE(!isScalar(i), "[JIT] Cannot have output or pointer-value cpu scalars on cuda kernels");
            }
            if (is_const[i]) {
                MEGU_ENSURE(i < operands_.size(), "[JIT]Argument mismatch, parsed arg",
                    i, "  but only ", operands_.size(), " were given."); 
                MEGU_ENSURE(operands_[i].exists(), "[JIT] Only outputs are allowed to be undefined");
            }
            ++i;
        }
        MEGU_ENSURE(i == operands_.size(), "[JIT]Argument mismatch, parsed ",
            i, " arguements  but ", operands_.size(), " were given.");

        MEGU_ENSURE(noutputs != 0, "[JIT] Operation with no output parameters is not allowed (and doesnt even make sense duh)");
    }

    ShapeVector JitDeez::getBroadcastShape() const
    {
        ShapeVector outshape;
        if (operands_.size() == 1)
        {
            outshape = ShapeVector(operands_[0].shape());
            return outshape;
        }
        int fallback_outshape_idx = -1;
        int fallback_inshape_idx = -1;
        int out_dim = -1;
        SmallVector<int,5> indexed ;
        for (int i = 0; i < operands_.size(); ++i)  {
            /*
            * 2 possible fastpaths
            */

            if (isOutput(i) && operands_[i].exists())
            {
                if (!isPointer(i)) {
                    outshape = ShapeVector(operands_[i].shape());
                    return outshape;
                }
                else {
                    fallback_outshape_idx = i;
                }
            }
            if (isInput(i)) {
                fallback_inshape_idx = i;
            }
            if (isInput(i) && !isPointer(i) && !isScalar(i)) {
                indexed.push_back(i);
                out_dim = std::max(static_cast<int>(operands_[i].shape().size()), out_dim);
            }
            // if(!isScalar(i) && !isPointer(i) && operands_[i].exists())
            //   out_dim = std::max(static_cast<int>(operands_[i].shape().size()), out_dim); 

        }
        if (indexed.size() == 0) {
            if (fallback_outshape_idx != -1) {
                outshape = ShapeVector(operands_[fallback_outshape_idx].shape());
                return outshape;
            }
            if (fallback_inshape_idx != -1) {
                outshape = ShapeVector(operands_[fallback_outshape_idx].shape()); 
                return outshape;
            }
            MEGU_ENSURE(false, "[JIT] Could not infer a shape for the operation that may be because the operation has no input",
                " arrays and all the output arrays are not initialized");
        }
        if (indexed.size() == 2)
        {
            outshape = shape_tricks::broadcast_shapes(operands_[indexed[0]].shape(), operands_[indexed[1]].shape());
            return outshape;
        }
        //this is the way cupy does multi broadcasting in cython 
        //from cupy 
        /*
        Copyright(c) 2015 Preferred Infrastructure, Inc.
            Copyright(c) 2015 Preferred Networks, Inc.

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files(the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions :

        The above copyright notice and this permission notice shall be included in
            all copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
            THE SOFTWARE.
        */
        SmallVector<ShapeVector, 4> pad; 
        pad.resize(indexed.size()); 
        int ptr = 0;
        for (const int i : indexed) 
        {
            pad[ptr].resize(out_dim); 
            int dim = operands_[i].shape().size();  
            int s = out_dim - dim;  
            for (int ii = 0; ii < s; ++ii) 
                pad[ptr][ii] = 1ull; 
            for (int ii = 0; ii < dim; ++ii) 
                pad[ptr][ii + s] = operands_[i].udim(ii); 
            ++ptr; 
        };
        for (int i = 0; i < pad[0].size(); ++i) {
            bool flag = true;
            size_t d = 1;
            for (int ii = 0; ii < pad.size(); ++ii) {
                size_t el = pad[ii][i];
                d = flag && el != 1 ? (flag = false, el) : d;
            }
            for (int ii = 0; ii < pad.size(); ++ii) {
                MEGU_ENSURE(pad[ii][i] == 1 || pad[ii][i] == d,
                    "operands could not be broadcasted together ");
            }
            outshape.push_back(d);
        }
        return outshape;
    }
    

    JitFunctionArg JitDeez::setArgs(ElementWiseArgument& arg)
    {
        broadcastAndAllocate();

        auto [shape, keep] = squashShape();
        
        JitFunctionArg out(arg,device());

        arg.compute_type = compute_type_.value(); 
        arg.shape = shape; 

        out.is_const = is_const;  
        out.is_pointer = is_raw; 

        for (int i = 0; i < operands_.size();++i) {
            const auto& op = operands_[i];
            if (op.is_scalar())
            {
                out.is_scalar[i] = true;
                out.scalars.emplace_back(op.scalar());
                continue;
            }
            arg.dtype.emplace_back(op.dtype());   
            arg.data.emplace_back(op.data());    
            if (!isPointer(i)) {
                arg.stride.emplace_back(shape_tricks::get_squashed_strides(op.strides(), keep));  
            }
        }
        return out; 
    }
   

    //all the simplifying/squashing code is from https://github.com/chainer/chainer
        /*
        Copyright(c) 2015 Preferred Infrastructure, Inc.
            Copyright(c) 2015 Preferred Networks, Inc.

            Permission is hereby granted, free of charge, to any person obtaining a copy
            of this software and associated documentation files(the "Software"), to deal
            in the Software without restriction, including without limitation the rights
            to use, copy, modify, merge, publish, distribute, sublicense, and /or sell
            copies of the Software, and to permit persons to whom the Software is
            furnished to do so, subject to the following conditions :

        The above copyright notice and this permission notice shall be included in
            all copies or substantial portions of the Software.

            THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
            IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
            FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.IN NO EVENT SHALL THE
            AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
            LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
            OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
            THE SOFTWARE.
            */
    std::tuple<megu::ShapeVector, megu::Axes>
        JitDeez::squashShape() const {

        const ShapeVector& shape = shape_.value();

        ShapeVector squashed{};
        Axes keep{};

        switch (int8_t ndim = shape.size()) {
        case 0:
            break;
        case 1:
            squashed = shape;
            keep.emplace_back(0);
            break;
        default:
            // Create a temporary shape with equal number of dimensions as in the original shape, but that will hold 1s where axes later can
            // be squashed.
            ShapeVector compressed = shape;
            for (int8_t i = 1; i < ndim; ++i) {
                if (compressed[i - 1] == 1) {
                    // Do nothing.
                }
                else if (isSquashableDimension(i, compressed)) {
                    compressed[i] *= compressed[i - 1];
                    compressed[i - 1] = 1;
                }
                else {
                    keep.emplace_back(i - 1);
                }
            }
            if (compressed.back() != 1) {
                keep.emplace_back(ndim - 1);
            }

            if (keep.size() == ndim) {
                // No axes could be squashed.
                squashed = compressed;
                break;
            }
            // Squash compressed axes.
            std::copy_if(compressed.begin(), compressed.end(), std::back_inserter(squashed), [](int64_t dim) { return dim != 1; });
            break;
        }
        return std::make_tuple(squashed, keep);


    }
}