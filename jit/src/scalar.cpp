#include "../include/scalar.hpp"
#include "../common/visit.h"
#include "../common/macros.h"

namespace megu {
    bool Scalar::is_zero() const {
        bool x = 0;
        MEGU_VISIT_DTYPE(
            type, scalar_t,
            x = (*reinterpret_cast<const scalar_t*>(value) == scalar_t(0));
        break;
        );
        return x;
    }

    Scalar Scalar::neg() const {
        Scalar ret(0);
        if (type == Bool)
        {
            ret.type = Int64;
        }
        else
        {
            ret.type = type;
        }
        MEGU_VISIT_NUMERICS(
            ret.dtype(), scalar_t,
            scalar_t val = *this;
            val = -val;
            new(ret.value) scalar_t(val);
        )
            return ret;
    }

    Scalar Scalar::reciprocal() const {
        Scalar ret(0);

        if (detail::is_floating_point(dtype()) || detail::is_complex(dtype()))
        {
            ret.type = dtype();
        }
        else {
            ret.type = Float;
        }
        MEGU_VISIT_FLOATS_COMPLEX_AND2(
            ret.type, scalar_t, Half, BFloat16,

            scalar_t val = *this;
        new(ret.value) scalar_t(scalar_t(1) / val);
        )
            return ret;
    }

    Scalar Scalar::astype(dtype_t t)const {
        MEGU_VISIT_DTYPE(dtype(), T,
            T _t = *this;
        return _t;
        );
    }

    Scalar Scalar::conj()const {
        if (detail::is_complex(dtype())) {
            if (dtype() == Complex64) {
                std::complex<float> _t = *this;
                return std::conj(_t);
            }
            else {
                std::complex<double> _t = *this;
                return std::conj(_t);
            }
        }
        return *this;
    }

}//end megu