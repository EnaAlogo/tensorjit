#include "../include/codegen.hpp"

#include "../include/shape_vector.hpp"
#include "../include/jit_function_arg.hpp"
#include "../common/array_ref.hpp"
#include "../common/string_util.hpp"
#include "jitcode_strings.h"
#include "../common/memory_ops.h"
#include "reduction_code.h"

namespace megu::cuda::jit
{
    const std::string pointwise_kernel_template =
        R"ESC(${name}
#define N_IDXED ${n_idxd}
#define NARGS ${nargs}

#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <math.h>

${includes}

)ESC"
R"ESC(

#define INDEX_T ${index_type}

  __global__ void ${kernel_name} (megu::detail::Carray<void*,NARGS> __data ,
                                  megu::detail::${calc} _iC , const INDEX_T __numel  ${scalar_args} ){
       for (INDEX_T __idx = blockIdx.x * blockDim.x + threadIdx.x; __idx < __numel; __idx += blockDim.x * gridDim.x){
          const auto __offsets = _iC.get(__idx);
           
          ${vars}
          ${body};
       }
   };
)ESC";

     


    const std::string vectorized_kernel_template =
        R"ESC(${name}
#define N_IDXED ${n_idxd}
#define NARGS ${nargs}
#if defined(__HIPCC__)
#define NUM_THREADS 256
#else
#define NUM_THREADS 128
#endif
#define THREAD_WORK_SIZE 4
constexpr int BLOCK_WORK_SIZE = THREAD_WORK_SIZE*NUM_THREADS;


#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <math.h>

${includes}

)ESC"
R"ESCAPE(
template<typename scalar_t, int vec_size>
struct alignas(sizeof(scalar_t)* vec_size) gpu_vec {
	scalar_t val[vec_size];
};


template <int vec_size, typename scalar_t ,typename index_t = unsigned int>
inline gpu_vec<scalar_t, vec_size> vload(const scalar_t* base_ptr, index_t offset) {
     using vec_t = gpu_vec<scalar_t, vec_size>;
     auto* from = reinterpret_cast<const vec_t*>(base_ptr);
     return from[offset];
}

template <int vec_size , typename index_t = unsigned int> 
inline gpu_vec<bool, vec_size> vload(const bool* base_ptr, index_t offset) {
   auto tmp = vload<vec_size,uint8_t , index_t>(reinterpret_cast<const uint8_t*>(base_ptr), offset); 
   gpu_vec<bool, vec_size> ret;
   for (int i = 0; i < vec_size; ++i) {
       ret.val[i] = bool(tmp.val[i]);
   }
   return ret;
}

)ESCAPE"
R"ESCAPE(  


#define VEC_SIZE ${vec_size}
#define INDEX_T ${index_type}

__global__ void ${kernel_name} (megu::detail::Carray<void*,NARGS> __data , const INDEX_T __numel  ${scalar_args})  {
      const INDEX_T remaining = __numel - BLOCK_WORK_SIZE*blockIdx.x;
      int thread_idx = threadIdx.x;

      ${vars}

      if (remaining < BLOCK_WORK_SIZE) {
        #pragma unroll
        for (int j = 0; j < THREAD_WORK_SIZE; j++){
          if (thread_idx >= remaining) {
            break;
          }
          const int __idx = thread_idx + BLOCK_WORK_SIZE*blockIdx.x;
          ${load}
           
          thread_idx += NUM_THREADS;
        }
        #pragma unroll
        for (int j = 0; j < THREAD_WORK_SIZE; j++) {
          if ((threadIdx.x  + j*NUM_THREADS) < remaining) {
            ${body};
          }
        }
        thread_idx = threadIdx.x;
        #pragma unroll
        for (int j = 0; j < THREAD_WORK_SIZE; j++) {
          if (thread_idx >= remaining) {
              break;
          }
          const int __idx = thread_idx + BLOCK_WORK_SIZE*blockIdx.x;
          ${store}
          thread_idx += NUM_THREADS;
        }
      } else {
        static constexpr int LOOP_SIZE = THREAD_WORK_SIZE/VEC_SIZE;
        
        ${invecs}
         
        #pragma unroll
        for (int i = 0; i<LOOP_SIZE; i++){
        
          ${vload}
          thread_idx += NUM_THREADS;
        }
        #pragma unroll
        for (int j = 0; j < THREAD_WORK_SIZE; j++) {
          ${body};
        }

        ${outvecs}

        thread_idx = threadIdx.x;
        #pragma unroll
        for (int i = 0; i<LOOP_SIZE; i++){
          ${vstore}
          thread_idx += NUM_THREADS;
        }
     }
}
)ESCAPE";


#define step(key,value) \
    idx = s.find(key, curr);\
    ss << std::string_view{ s.data() + curr  ,idx - curr}; \
    ss << value;\
    curr = idx + sizeof(key) - 1

    typedef enum
    {
        NO_SUPPORT,
        PARTIAL,
        FULL
    }Support_t;

    namespace impl {
        template<typename F>
        static inline Support_t _computeSupportGeneric(const JitFunctionArg& arg, const F& f)
        {
            if (f(arg.arg.compute_type))
            {
                return Support_t::FULL;
            }

            for (int i = 0; i < arg.nargs(); ++i)
            {
                if (f(arg.arg.dtype[i]))
                    return Support_t::PARTIAL;
            }
            for (int i = 0; i < arg.scalars.size(); ++i)
            {
                if (f(arg.scalars[i].dtype()))
                    return Support_t::PARTIAL;
            }

            return Support_t::NO_SUPPORT;

        }
    }//end detail

    static inline Support_t needsComplexSupport(const JitFunctionArg& arg)
    {
        return impl::_computeSupportGeneric(arg, &megu::detail::is_complex);
    }

    static inline Support_t needsHalfSupport(const JitFunctionArg& arg)
    {
        return impl::_computeSupportGeneric(arg, [](dtype_t t) {return t == megu::Half; });
    }


    static inline Support_t needsBfloatSupport(const JitFunctionArg& arg)
    {
        return impl::_computeSupportGeneric(arg, [](dtype_t t) {return t == megu::BFloat16; });
    }

    static inline constexpr std::string_view getVarName(std::string_view arg)
    {
        int64_t pos = arg.size() - 1;
        for (; pos > -1 && !std::isspace(arg[pos]) && arg[pos] != '*'; --pos);
        ++pos;
        return std::string_view{ arg.begin() + pos , arg.end()};
    }

    static inline
        std::string getLoads(
            const cuda::jit::JitFunctionArg& arg,
            SmallVector<int8_t, 10>const& abs_in_indices,
            SmallVector<int8_t, 10>const& rel_in_indices)
    {
        std::ostringstream ss;
        for (int i = 0; i < rel_in_indices.size(); ++i)
        {
            ss << "\n          __in_vec" << i << "[j] = *(reinterpret_cast<"
                << arg.get_typename(rel_in_indices[i]) << " const*>(__data[" <<
                int(rel_in_indices[i]) << "]) + __idx);";
        }
        return ss.str();
    }

    static inline
        std::string getStores(
            const cuda::jit::JitFunctionArg& arg,
            SmallVector<int8_t, 10>const& abs_out_indices,
            SmallVector<int8_t, 10>const& rel_out_indices)
    {
        std::ostringstream ss;
        for (int i = 0; i < rel_out_indices.size(); ++i)
        {
            ss << "\n          *(reinterpret_cast<" << arg.get_typename(rel_out_indices[i]) << "*>(__data[" <<
                int(rel_out_indices[i]) << "]) + __idx) = __ot_vec" << i << "[j];";
        }
        return ss.str();
    }



    static inline
        std::string declareVectors(
            const cuda::jit::JitFunctionArg& arg,
            ArrayRef<std::string> names,
            SmallVector<int8_t, 10>& abs_in_indices,
            SmallVector<int8_t, 10>& abs_out_indices,
            SmallVector<int8_t, 10>& rel_in_indices,
            SmallVector<int8_t, 10>& rel_out_indices,
            int vec_size)
    {
        std::ostringstream out;
        int nargs = arg.scalars.size() + arg.arg.data.size(); 
        int ptr = 0;
        int noutptr = 0;
        int ninptr = 0;
        for (int i = 0; i < nargs; ++i)
        {
            if (arg.is_scalar[i])
                continue;
            if (arg.is_pointer[i])
            {
                out << "\n      auto *"
                    << getVarName(names[i])
                    << " = " << "reinterpret_cast<" << arg.get_typename(ptr);
                if (arg.is_const[i])
                    out << " const*";
                else
                    out << "*";
                out << ">(__data[" << ptr << "]);";
                ++ptr;
                continue;
            }
            const bool input = arg.is_input(i);
            out << "\n      "<<arg.get_typename(ptr) << " ";
            out << (input ? "__in_vec" : "__ot_vec")  <<
                (input ? ninptr++ : noutptr++) << '[' << vec_size << "];";
            if (input) {
                abs_in_indices.push_back(i);
                rel_in_indices.push_back(ptr);
            }
            else {
                abs_out_indices.push_back(i);
                rel_out_indices.push_back(ptr);
            }

            ++ptr;
        }
        return out.str();
    }

    static inline
    std::string getDoComputeBody(const cuda::jit::JitFunctionArg& arg,
        ArrayRef<std::string> names,
        SmallVector<int8_t, 10> const& abs_inputs,
        SmallVector<int8_t, 10> const& abs_outputs,
        SmallVector<int8_t, 10> const& rel_inputs,
        SmallVector<int8_t, 10> const& rel_outputs,
        std::string_view user_code)
    {
        std::ostringstream out;
        for (int i = 0; i < abs_inputs.size(); ++i)
        {
            int index = abs_inputs[i];

            std::string_view name = getVarName(names[index]);

            out << "\n          ";
            out << arg.get_typename(rel_inputs[i]) << " const& " << name
                << "  = __in_vec"<<i<<"[j];";
        }
        for (int i = 0; i < abs_outputs.size(); ++i)
        {
            int index = abs_outputs[i];
            std::string_view name = getVarName(names[index]);

            out << "\n          ";
            out << arg.get_typename(rel_outputs[i]) << "& " << name
                << " = __ot_vec"<<i<<"[j];";
        }
        out << "\n          ";
        out << user_code;
        return out.str();
    }
    static inline
        std::string getLoadSIMD(const cuda::jit::JitFunctionArg& arg,
            SmallVector<int8_t, 10> const& abs_inputs,
            SmallVector<int8_t, 10> const& rel_inputs)
    {
        std::ostringstream ss;
        for (int i = 0; i < rel_inputs.size(); ++i) {
            ss << "\n          const auto __vin_" << i << " = vload<VEC_SIZE>(src_" << i
                << ",thread_idx);";
        }
        ss << "\n\n          ";
        ss << "#pragma unroll\n          for(int j=0;j<VEC_SIZE;++j){";
        for (int i = 0; i < rel_inputs.size(); ++i) {
            ss <<"\n            "
                << "__in_vec" << i << "[VEC_SIZE*i +j] = __vin_" << i << ".val[j];";
        }
        ss << "\n          }\n";
        return ss.str();
    }
    static inline
        std::string getStoreSIMD(const cuda::jit::JitFunctionArg& arg,
            SmallVector<int8_t, 10> const& abs_outputs,
            SmallVector<int8_t, 10> const& rel_outputs)
    {
        std::ostringstream ss;
        for (int i = 0; i < abs_outputs.size(); ++i) {
            ss << "\n          gpu_vec<" 
                << arg.get_typename(rel_outputs[i]) << ",VEC_SIZE> v_"
                << i << ';';
        }
        ss << "\n          #pragma unroll\n          for(int j=0;j<VEC_SIZE;++j){";
        for (int i = 0; i < abs_outputs.size(); ++i) {
            ss << "\n            v_" << i << ".val[j] = __ot_vec"<<i<<"[VEC_SIZE*i + j];";
        }
        ss << "\n          }\n";
        for (int i = 0; i < abs_outputs.size(); ++i) {
            ss << "\n          dest_" << i << "[thread_idx] = v_" << i << ';';
        }
        return ss.str();
    }
    static inline
        std::string getOutVectors(const cuda::jit::JitFunctionArg& arg,
            SmallVector<int8_t, 10> const& abs_outputs,
            SmallVector<int8_t, 10> const& rel_outputs) {

        std::ostringstream ss;

        for (int i = 0; i < rel_outputs.size(); ++i)
        {
            ss << "\n        auto* dest_" << i << " = reinterpret_cast<gpu_vec<" << arg.get_typename(rel_outputs[i])
                << ",VEC_SIZE>*>(__data[" << int(rel_outputs[i]) << "]) + BLOCK_WORK_SIZE / VEC_SIZE *blockIdx.x;";
        }
        return ss.str();

    }
    static inline
        std::string getInVectors(const cuda::jit::JitFunctionArg& arg,
            SmallVector<int8_t, 10> const& abs_inputs,
            SmallVector<int8_t, 10> const& rel_inputs) {

        std::ostringstream ss;
        
        for (int i = 0; i < rel_inputs.size(); ++i)
        {
            ss << "\n        auto* src_" << i << " = reinterpret_cast<" << arg.get_typename(rel_inputs[i])
                << " const*>(__data[" << int(rel_inputs[i]) << "]) + BLOCK_WORK_SIZE*blockIdx.x;";
        }
        return ss.str();

    }
    static inline std::size_t get_includes(
        std::ostringstream& ss,
        std::size_t prev,
        const JitFunctionArg& arg,
        std::string const& s,//kernel
        const bool needs_index = true
    )
    {
        std::size_t idx = s.find("${includes}", prev);
        ss << std::string_view{ s.data() + prev , idx - prev };
        if (needs_index) { 
            ss << indexer_code;
        }
        if (needsBfloatSupport(arg) != Support_t::NO_SUPPORT)
        {
            ss << bfloat_code << bfloat16_math_code;
        }
        if (needsHalfSupport(arg) != Support_t::NO_SUPPORT)
        {
            ss << float16_code << float16_math_code;
        }
        if (needsComplexSupport(arg) != Support_t::NO_SUPPORT)
        {//temporary
            
         //does thrust have their own macro for this? couldnt find anything besiedes utils like THRUST_STD_COMPLEX_DEVICE
            ss << "\n#include <thrust/complex.h>\nusing namespace thrust;\n#define MEGU_COMPLEX_H\n"
                "namespace megu{"
                "template<typename T>"
                "inline bool IsNan(thrust::complex<T> val) {"
                "return isnan(val.real()) || isnan(val.imag());}}"; 
        }
        return idx + sizeof("${includes}") - 1;
    }

    static inline std::string get_includes(
        dtype_t inType,
        dtype_t outType,
        dtype_t computeType
        )
    {
        auto const isAnyOf = [](dtype_t in, ArrayRef<dtype_t> of) ->bool {
            for (auto d : of) {
                if (in == d)
                    return true;
            }
            return false;
        };
        std::stringstream ss; 
        if (isAnyOf(BFloat16 , {inType,outType,computeType}) ) {
            ss << bfloat_code << bfloat16_math_code;
        }
        if (isAnyOf(Half,{inType,outType,computeType})){
            ss << float16_code << float16_math_code;
        }
        if (isAnyOf(Complex64,{inType,outType,computeType}) || isAnyOf(Complex128,{inType,outType}))
        {//temporary

         //does thrust have their own macro for this? couldnt find anything besiedes utils like THRUST_STD_COMPLEX_DEVICE
            ss << "\n#include <thrust/complex.h>\nusing namespace thrust;\n#define MEGU_COMPLEX_H";
        }
        return ss.str();
    }

    static inline void define_scalar_args(
        std::ostringstream& ss,
        std::size_t& cpos,
        JitFunctionArg const& arg,
        ArrayRef<std::string>names,
        const std::bitset<24> is_scalar, 
        const std::string& s//kernel 
    )
    {
        int nargs = arg.arg.dtype.size() + arg.scalars.size(); 
        {
            auto idx = s.find("${scalar_args}", cpos);
            ss << std::string_view{ s.data() + cpos ,idx - cpos };
            int scalar_ptr = 0;
            for (int i = 0; i < nargs; ++i) {
                if (!is_scalar[i])
                    continue;
                ss << ",const " << arg.get_C_name(arg.scalars[scalar_ptr++].dtype()) << " "; 
                ss << getVarName(names[i]); 
            }
            cpos = idx + sizeof("${scalar_args}") - 1;
        }
    }

    static inline void get_kernel_body(
        std::ostringstream& ss,
        std::size_t cpos,
        const std::string_view user_code,
        const JitFunctionArg& arg,
        ArrayRef<std::string> names,
        const std::bitset<24> is_ptr,
        const std::bitset<24> is_const,
        const std::bitset<24>  is_scalar
    )
    {
        std::string const& s = pointwise_kernel_template;
        int const nargs = arg.scalars.size() + arg.arg.data.size();

        define_scalar_args(ss, cpos, arg, names, is_scalar, pointwise_kernel_template);

        std::size_t prev = cpos;
        cpos = s.find("${vars}", cpos);
        ss << std::string_view{ s.data() + prev ,  cpos - prev };
        cpos += sizeof("${vars}") - 1;
        int n_non_ptr = 0;
        int non_scalar_ptr = 0;
        for (int i = 0; i < nargs; ++i)
        {
            if (is_scalar[i])
                continue;
            const std::string_view dtype = arg.get_typename(non_scalar_ptr);
            ss << dtype;
            if (is_const[i]) {
                ss << " const";
            }
            if (is_ptr[i]) {
                ss << "* ";
            }
            else {
                ss << "& ";
            }
            ss << getVarName(names[i])<< " = ";
            ss << "reinterpret_cast<"
                << dtype;
            if (is_const[i])
            {
                ss << " const";
            }
            ss << "*>(__data["
                << non_scalar_ptr << "])";
            if (!is_ptr[i]) {
                ss << "[__offsets[" << n_non_ptr++ << "]]";
            }
            ss << ";\n          ";
            non_scalar_ptr++;
        }

        auto idx = s.find("${body}", cpos);
        ss << user_code;
        ss << std::string_view{ s.data() + idx + sizeof("${body}") - 1 , s.size() - (idx + sizeof("${body}") - 1) };
    }
    static inline std::size_t get_kernel_def(
        std::ostringstream& ss,
        const std::string_view name,
        ContentArgs args,
        const int nargs,
        std::size_t curr
    )
    {
        std::string const& s = pointwise_kernel_template;
        std::size_t idx;
        if (args.can_use_32bit_indexing) {
            step("${index_type}", "unsigned int");
        }
        else {
            step("${index_type}", "uint64_t");
        }
        step("${kernel_name}", name);
        ss << "_kernel";
        if (args.is_contiguous)
        {
            step("${calc}", "ContiguousIndexToOffset<");
            ss << nargs << ",INDEX_T>";
        }
        else {
            step("${calc}", "IndexToOffset<");
            ss << nargs << ",INDEX_T>";
        }
        return curr;
    }

    static inline std::size_t get_kernel_def_vec(
        std::ostringstream& ss,
        const std::string_view name,
        ContentArgs args,
        const int nargs,
        std::size_t curr,
        int vec_size
    )
    {
        std::string const& s = vectorized_kernel_template;
        std::size_t idx;
        if (args.can_use_32bit_indexing) {
            step("${index_type}", "unsigned int");
        }
        else {
            step("${index_type}", "uint64_t");
        }
        step("${kernel_name}", name);
        ss << "_kernel"<<"_vectorized"<<vec_size; 

        return curr;
    }

    static inline std::size_t get_kernel_preamble
    (
        std::ostringstream& ss,
        const int nargs,
        const int n_idx,
        const std::string_view name,
        std::string const& s//kernel(vectorized of grid strided)
    )
    {
        std::size_t curr = 0;
        std::size_t idx;

        step("${name}", name); 
        step("${n_idxd}", n_idx);
        step("${nargs}", nargs);
        ss << preamble_code;
        return curr;
    }

    static inline
        std::string codegen_nonvec(
            JitFunction const& f,
            JitFunctionArg const& arg,
            ContentArgs details)
    {
        std::ostringstream ss;
        const int nidx = arg.nargs() - arg.is_pointer.count();
        std::size_t pos = get_kernel_preamble(ss, arg.nargs(), nidx, f.function_name() , pointwise_kernel_template); 
        pos = get_includes(ss, pos, arg , pointwise_kernel_template);  
        pos = get_kernel_def(ss, f.function_name(), details, nidx, pos);
        get_kernel_body(ss, pos, f.function_body(), arg, f.function_args(), arg.is_pointer, arg.is_const, arg.is_scalar);
        return ss.str();
    }

    static inline
    std::string codegen_vectorized(
        JitFunction const& f,
        JitFunctionArg const& arg,
        ContentArgs details,
        int vec_size
    )
    {
        std::ostringstream ss;
        const int nidx = arg.nargs() - arg.is_pointer.count();
        std::size_t curr = get_kernel_preamble(ss, arg.nargs(), nidx, f.function_name(), vectorized_kernel_template);
        curr = get_includes(ss, curr, arg ,vectorized_kernel_template,false);

        std::size_t idx;

        SmallVector<int8_t, 10> absin, absout, relin, relout;

        std::string const decls = declareVectors(arg, f.function_args(), absin, absout, relin, relout, vec_size);

        std::string const& s = vectorized_kernel_template;

        std::string const body = getDoComputeBody(arg, f.function_args(), absin, absout, relin, relout, f.function_body());

        step("${vec_size}", vec_size);

        curr = get_kernel_def_vec(ss, f.function_name(), details, arg.scalars.size() + arg.arg.data.size(),
            curr, vec_size);

        define_scalar_args(ss, curr, arg, f.function_args(), arg.is_scalar, vectorized_kernel_template);

        step("${vars}", decls);

        step("${load}", getLoads(arg, absin, relin));

        step("${body}", body);

        step("${store}", getStores(arg, absout, relout));

        step("${invecs}", getInVectors(arg, absin, relin));

        step("${vload}", getLoadSIMD(arg, absin, relin)); 

        step("${body}", body);

        step("${outvecs}", getOutVectors(arg, absout, relout));

        step("${vstore}", getStoreSIMD(arg, absout, relout));

        ss << s.substr(curr); 

        return ss.str();  

    }
#undef step

    std::string codegen(
        JitFunction const& f,
        JitFunctionArg const& arg,
        ContentArgs const& details
    )
    {
        if (details.vectorization.vectorized) {  
            return codegen_vectorized(f, arg, details, details.vectorization.vec_size); 
        }

        return codegen_nonvec(f, arg, details);
    }

    std::string format_code(std::string_view code) {
        int i = 0;
        std::ostringstream ss;
        for (auto line : detail::split(code, '\n')) {
            ss << i++ << ": " << line << '\n';
        }
        return ss.str();
    }

    std::string codegen_reduction(std::string_view impl,
        std::string_view kernel_name,
        int max_threads , 
        int vt0 ,
        int output_vec_size,
        bool can_use32bit,
        dtype_t in_dtype,
        dtype_t out_type,
        dtype_t compute_type,
        std::optional<JitScalarArg>const& scalar )
    {
        std::string const pre = preamble_code + indexer_code + get_includes(in_dtype,out_type, compute_type); 
        auto code = detail::replace_first(reduction_template, "${impl}", impl);
        detail::replace_first_(code, "${max_threads_lb}", std::to_string(max_threads));
        detail::replace_first_(code, "${vt0}", std::to_string(vt0)); 
        detail::replace_first_(code, "${kernel_name}", kernel_name);
        detail::replace_first_(code, "${program_name}", std::string(kernel_name)+"_program"); 
        detail::replace_first_(code, "${inT}", JitFunctionArg::get_C_name(in_dtype)); 
        detail::replace_first_(code, "${outT}", JitFunctionArg::get_C_name(out_type)); 
        detail::replace_first_(code, "${indexT}", can_use32bit ? "unsigned int" : "uint64_t");
        detail::replace_first_(code, "${output_vec_size}", std::to_string(output_vec_size));
        detail::replace_first_(code, "${preamble}", pre);
        if (scalar) {
            std::ostringstream scalar_decl;
            scalar_decl << JitFunctionArg::get_C_name(scalar->dtype());
            scalar_decl << " const " << scalar->name() << ";";
            detail::replace_first_(code, "${scalars}", scalar_decl.str());
        }
        else {
            detail::replace_first_(code, "${scalars}", "");
        }
        return code;
    }
    

    std::string simple_reduction_impl_code(
        std::string_view const combine,
        std::optional<std::string_view> identity ,
        std::optional<std::string_view> helper_functions,
        std::optional<std::string_view> reduce,
        std::optional<std::string_view> project,
        std::optional<std::string_view> interm_type) {
        using namespace std::string_view_literals;


        std::string_view const c =
            "using arg_t = ${marg_t} ;\n"
            "#define ident ${identity}\n"
            "${helpers}\n"
            "${combine}\n"
            "${project}\n"
            "static inline __device__ arg_t warp_shfl_down(arg_t arg, int offset) {return WARP_SHFL_DOWN(arg, offset);}\n"
            "static inline __device__ arg_t translate_idx(arg_t acc, int64_t /*idx*/) {return acc;}\n"
            "${reduce}"sv; 

        auto out = detail::replace_first(c, "${combine}", combine);
        detail::replace_first_(out, "${reduce}", 
            reduce.value_or("static inline __device__ arg_t reduce(arg_t acc, scalar_t val, int64_t idx){return combine(acc, val);}\n"sv)
            );
        detail::replace_first_(out, "${project}",
            project.value_or("static inline __device__ out_scalar_t project(arg_t arg) { return (out_scalar_t)arg; }\n"sv)
        );

        detail::replace_first_(out, "${identity}", identity.value_or("(scalar_t)0"));
        detail::replace_first_(out, "${helpers}", helper_functions.value_or("")); 
        detail::replace_first_(out, "${marg_t}", interm_type.value_or("megu::acctype_t<scalar_t>"));
        return out; 
    }


}//end megu::cuda::jit