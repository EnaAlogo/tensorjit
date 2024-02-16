#include "../common/shape_tricks.hpp" 
#include "../include/tensor.h"

namespace megu::shape_tricks {




    ShapeVector broadcast_stride(const JitTensor& self, LongArrayView broadcasted_shape) {
        return broadcast_stride(self.shape(), self.strides(), std::move(broadcasted_shape));
    }

    std::string stringify_sizes(ArrayRef<JitTensor> tensr) {
        std::stringstream ss;
        for (const auto& t : tensr)
            ss << t.shape() << " ";
        return ss.str();
    }

    void broadcast_inplace(MutableArrayRef<JitTensor>tensors)
    {
        auto shape = broadcast_shapes(tensors);
        for (auto& array : tensors) {
            if (array.data() && array.shape() != shape) {
                array.broadcast_inplace(shape);
            }
        }
    }

    ShapeVector broadcast_shapes(ArrayRef<JitTensor> tensr) {

        ShapeVector outshape;

        if (tensr.size() == 1)
        {
            outshape = ShapeVector(tensr[0].shape());
            return outshape;
        }
        else if (tensr.size() == 2)
        {
            outshape = broadcast_shapes(tensr[0].shape(), tensr[1].shape());
            return outshape;
        }

        int out_dim = -1;
        for (int i = 0; i < tensr.size(); ++i)
            out_dim = std::max(static_cast<int>(tensr[i].rank()), out_dim);

        SmallVector<ShapeVector, 4> pad;
        pad.resize(tensr.size());
        for (int i = 0; i < tensr.size(); ++i) {
            pad[i].resize(out_dim);
            int dim = tensr[i].rank();
            int s = out_dim - dim;
            for (int ii = 0; ii < s; ++ii)
                pad[i][ii] = 1ull;
            for (int ii = 0; ii < dim; ++ii)
                pad[i][ii + s] = tensr[i].udim(ii);
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
                    "operands could not be broadcasted together ",
                    "with shapes ", stringify_sizes(tensr));
            }
            outshape.push_back(d);
        }
        return outshape;

    }
}//end megu::shape_tricks