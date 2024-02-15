#pragma once
#include "shape_vector.hpp"
#include "../common/array_ref.hpp"

namespace megu::memory {

    enum memory_format_t : char {
        ROW_MAJOR,// typical C contiguous
        COL_MAJOR,// Fortran contiguous
        CHANNELS_LAST,// similar to mkldnn and torch channels last format where the shape is NCHW and the stride of C is 1 
        STRIDED, // custom strides eg when a tensor is broadcasted and has 0 stride dims
        CF_CONTIGUOUS, // rare case where its both C and Fortran contiguous eg a vector , a (4,1) matrix etc

        NUM_FORMATS // the number of available formats
    };

    inline const char* formatName(memory_format_t format)
    {
        switch (format)
        {
        case megu::memory::ROW_MAJOR:
            return "Row Major";
        case megu::memory::COL_MAJOR:
            return "Col Major";
        case megu::memory::STRIDED:
            return "Strided";
        case megu::memory::CF_CONTIGUOUS:
            return "CF Contiguous";
        case megu::memory::CHANNELS_LAST:
            return "Channels First";
        default:
            MEGU_ENSURE(0, "Could not resolve format ", std::to_string(int(format)) );
        }
    }

    template<class outIt, class It>
    constexpr void computeContiguousStrides(outIt strides, It begin, It end,
        size_t trail_stride = 1)
    {
        std::ptrdiff_t size = end - begin;
        if (size <= 0)return;
        strides[size - 1] = trail_stride;
        for (std::ptrdiff_t i = size - 1; i > 0; --i) {
            strides[i - 1] = strides[i] * (*(begin + i));
        }
    }

    inline ShapeVector computeContiguousStrides(LongArrayView vec)
    {
        ShapeVector out(vec.size());
        computeContiguousStrides(out.data(), vec.begin(), vec.end(), 1);
        return out;
    }


    inline ShapeVector computeFortranContiguousStrides(LongArrayView vec)
    {
        ShapeVector out(vec.size());
        computeContiguousStrides(out.rbegin(), vec.rbegin(), vec.rend(), 1);
        return out;
    }

    inline void computeChannelsLastStrides(int64_t* strides, LongArrayView sizes) {
        switch (sizes.size()) {
            case 3:
                strides[1] = 1;
                strides[2] = sizes[1];
                strides[0] = strides[2] * sizes[2];
                break;
            case 4:
                strides[1] = 1;
                strides[3] = sizes[1];
                strides[2] = strides[3] * sizes[3];
                strides[0] = strides[2] * sizes[2];
                break;
            case 5:
                strides[1] = 1;
                strides[4] = sizes[1];
                strides[3] = strides[4] * sizes[4];
                strides[2] = strides[3] * sizes[3];
                strides[0] = strides[2] * sizes[2];
                break;
            default:
                MEGU_ENSURE(false, "Channels Last format is not implemented for ", std::to_string(sizes.size()), " dimensions ,",
                    "if you are using a single image without the batch dimension make sure you prepend a 1 before calling ",
                    "this function");
            }
        }
        // always assumes the first dimension is the batch
        inline ShapeVector computeChannelsLastStrides(LongArrayView sizes) {
            ShapeVector strides(sizes.size());
            switch (sizes.size()){ 
                case 3:
                    strides[1] = 1; 
                    strides[2] = sizes[1];
                    strides[0] = strides[2] * sizes[2];
                    break;
                case 4:
                    strides[1] = 1; 
                    strides[3] = sizes[1];   
                    strides[2] = strides[3] * sizes[3];  
                    strides[0] = strides[2] * sizes[2];  
                    break;
                case 5:
                    strides[1] = 1; 
                    strides[4] = sizes[1]; 
                    strides[3] = strides[4] * sizes[4]; 
                    strides[2] = strides[3] * sizes[3]; 
                    strides[0] = strides[2] * sizes[2]; 
                    break;
                default:
                    MEGU_ENSURE(false, "Channels Last format is not implemented for ",std::to_string(sizes.size()), " dimensions ,",
                        "if you are using a single image without the batch dimension make sure you prepend a 1 before calling ",
                        "this function");
            }
            return strides;
        }

        template<typename InputInterator>
        inline void computeStrides(int64_t* stride, InputInterator begin , InputInterator end , memory_format_t format)
        {
            switch (format) {
            case ROW_MAJOR:
                return computeContiguousStrides(stride , begin  , end); 
            case COL_MAJOR:
                return computeContiguousStrides(
                    std::make_reverse_iterator(stride +(end-begin)) ,
                    std::make_reverse_iterator(end)  ,
                    std::make_reverse_iterator(begin));
            case CHANNELS_LAST:
                return computeChannelsLastStrides(stride, LongArrayView(begin, end));
            default:
                MEGU_ENSURE(0, "Unsupported format ", formatName(format)); 
            }
        }

        inline ShapeVector computeStrides(LongArrayView shape, memory_format_t format)
        {
            switch (format) {
            case ROW_MAJOR:
                return computeContiguousStrides(shape);
            case COL_MAJOR:
                return computeFortranContiguousStrides(shape);
            case CHANNELS_LAST:
                return computeChannelsLastStrides(shape); 
            default:
                MEGU_ENSURE(0, "Unsupported format ", formatName(format));
            }
        }

       

        inline bool isChannelsLastContiguous3D(
            const LongArrayView shape,
            const LongArrayView strides) {
            int64_t min = 0;
            if (strides[1] == 0) {
                return false;
            }
            for (auto& d : { 1, 4, 3, 2, 0 }) {
                if (shape[d] == 0) {
                    return false;
                }
                if (strides[d] < min) {
                    return false;
                }
                if (d == 0 && min == strides[1]) {
                    return false;
                }
                min = strides[d];
                if (shape[d] > 1) {
                    min *= shape[d];
                }
            }
            return true;
        }

        inline bool isChannelsLastContiguous2D(
            const LongArrayView shape,
            const LongArrayView strides)
        {
            int64_t min = 0;
            if (strides[1] == 0) {
                return false;
            }
            for (auto& d : { 1, 3, 2, 0 }) {
                if (shape[d] == 0) {
                    return false;
                }
                if (strides[d] < min) {
                    return false;
                }
                if (d == 0 && min == strides[1]) {
                    return false;
                }
                min = strides[d];
                if (shape[d] > 1) {
                    min *= shape[d];
                }
            }
            return true;
        }

        inline bool isChannelsLastContiguous(
            const LongArrayView shape,
            const LongArrayView strides)
        {
            assert(shape.size() == strides.size());
            switch (shape.size())
            {
            case 4://2d
                return isChannelsLastContiguous2D(shape, strides);
            case 5://3d
                return isChannelsLastContiguous3D(shape, strides);
            default:
                MEGU_ENSURE(false,"Only 4D or 5D tensors supported for nhwc");
            }
        }

        inline bool isDefaultContiguous(LongArrayView shape, LongArrayView stride) {
            assert(shape.size() == stride.size());
            int size = shape.size();
            int64_t step = 1;
            for (int i = size - 1; i > -1; --i) {
                const int64_t dim = shape[i];
                if (dim != 1) {
                    if (stride[i] == step)
                        step *= dim;
                    else
                        return false;
                }
            }
            return true;
        }

        inline bool isFortranContiguous(LongArrayView _shape, LongArrayView _stride) 
        {
            auto shape = _shape.rbegin();
            auto stride = _stride.rbegin(); 
            assert(_shape.size() == _stride.size()); 
            int size = _shape.size(); 
            int64_t step = 1;
            for (int i = size - 1; i > -1; --i) {
                const int64_t dim = shape[i];
                if (dim != 1) {
                    if (stride[i] == step)
                        step *= dim;
                    else
                        return false;
                }
            }
            return true;
        }

        inline bool isContiguous(LongArrayView shape, LongArrayView stride, memory_format_t format)
        {
            switch (format)
            {
            case megu::memory::ROW_MAJOR: 
                return isDefaultContiguous(shape, stride);  
            case megu::memory::COL_MAJOR: 
                return isFortranContiguous(shape, stride);
            case megu::memory::CHANNELS_LAST:
                return isChannelsLastContiguous(shape, stride);
            default:
                MEGU_ENSURE(0,"Unsupported format :  ",formatName(format)); 
            }
        }

        inline memory_format_t getMemoryFormat(LongArrayView shape, LongArrayView strides) {
            int dim = shape.size();
            if ((dim == 4 || dim == 5) && isChannelsLastContiguous(shape, strides)) 
                return CHANNELS_LAST; 

            auto rshape = shape.rbegin();
            auto rstrides = strides.rbegin();
            int64_t stepC = 1, stepF = 1;
            bool fortran = 1, c = 1;
            for (int i = dim - 1; i > -1; --i) {
                const int64_t fdim = rshape[i];
                const int64_t cdim = shape[i];
                if (cdim != 1) {
                    if (strides[i] == stepC)
                        stepC *= cdim;
                    else
                        c = false;
                }
                if (fdim != 1)
                {
                    if (rstrides[i] == stepF)
                        stepF *= fdim;
                    else
                        fortran = false;
                }

                if (!c && !fortran)
                    return STRIDED;
            }

            return fortran && c ? CF_CONTIGUOUS : fortran ? COL_MAJOR : ROW_MAJOR;
        }

        inline size_t byteSize(
            size_t typesize,
            LongArrayView shape,
            LongArrayView stride,
            int64_t offset)
        {
            size_t size = 1;
            for (int i = 0; i < shape.size(); ++i) {
                if (shape[i] == 0) return 0;
                
                size += stride[i] * (shape[i] - 1);
            }
            return typesize * (offset + size);
        }

      
}//end megu::memory