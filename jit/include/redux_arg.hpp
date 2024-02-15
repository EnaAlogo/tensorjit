#pragma once
#include "../common/kernel_dto.hpp" 
#include "../common/array_ref.hpp" 

namespace megu {


    struct ReduxArg : detail::KernelDTO<3> {


        int num_reduce_dims()const {
            int count = 0;
            for (auto dim = 0; dim < ndim(); ++dim) { 
                if (stride[0][dim] == 0) { 
                    count++;
                }
            }
            return count;
        }

        int64_t output_len()const {
            int64_t i = 1;
            for (int dim = 0; dim < ndim(); ++dim) {
                if (stride[0][dim] != 0 || shape[dim] == 0) {
                    i *= shape[dim];
                }
            }
            return i;
        }

        bool can_use_32bit_index() const {
            int64_t constexpr max_value = std::numeric_limits<int32_t>::max();
            if (len() > max_value) { 
                return false;
            }
            for (int op = 0; op < nargs() ; ++op) {
                int64_t max_offset = 1;
                for (int dim = 0; dim < ndim(); ++dim) {
                    max_offset += (shape[dim] - 1) * stride[op][dim];
                }
                if (max_offset > max_value) {
                    return false;
                }
            }
            return true;
        }
    };

}