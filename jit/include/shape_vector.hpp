#pragma once
#include "../common/macros.h"
#include <iostream>

#include "../common/small_vector.hpp"

namespace megu {

    constexpr int kMaxDimSize = 5;

    extern template class SmallVector<int, kMaxDimSize>;
    extern template class SmallVector<int64_t, kMaxDimSize>;

    using ShapeVector = SmallVector<int64_t, kMaxDimSize>;

    using Axes = SmallVector<int8_t, kMaxDimSize>;


 
};