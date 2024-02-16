#include "../include/jit_cuda.hpp"
#include "../common/string_util.hpp"
#include <jitify.hpp>
#undef max
#undef min

#include "../common/memory_ops.h"
#include "../include/codegen.hpp"
#include <variant>
#include "../common/cindexer.hpp"
#include <ranges>
#define CHECK_CUDA(_call)                                                  \
  do {auto call = _call;                                                               \
    if (call != CUDA_SUCCESS) {                                           \
      const char* str;                                                    \
      cuGetErrorName(call, &str);                                         \
      std::cout << "(CUDA) returned " << str;                             \
      std::cout << " (" << __FILE__ << ":" << __LINE__ << ":" << __func__ \
                << "())" << std::endl;                                    \
      return;                                                       \
    }                                                                     \
  } while (0)

namespace megu::cuda::jit
{
	static inline std::vector<std::string> get_args(std::string arg) {
		detail::trim_(arg);
		detail::remove_dup_chars_(arg, ' ');
		std::vector<std::string> out;
		std::ranges::split_view v(arg,',');
		for (auto const& s : v) {
			std::string s_{ std::string_view(s) };
			detail::trim_(s_); 
			if (!s_.empty()) {
				out.emplace_back(std::move(s_));
			}
		}
		return out;
	}

	JitFunction::JitFunction(std::string name, std::string args, std::string code)
		:name_(std::move(name)), args_(get_args(std::move(args))), body_(std::move(code))
	{
	}

	template<int args, typename index_t>
	static std::unique_ptr<megu::detail::IndexToOffset<args, index_t>> makeIndexer(JitFunctionArg const& arg) 
	{
		auto r = std::make_unique<megu::detail::IndexToOffset<args, index_t>>();
		r->dims = arg.arg.shape.size();
		for (int i = 0; i < r->dims; ++i) {
			r->shape[i] = IntDivider<index_t>(arg.arg.shape[i]);
			for (int j = 0; j < args; ++j) {
				r->strides_[i][j] = arg.arg.stride[j][i];
			}
		}
		return r;
	}
	template<int args, typename index_t>
	static megu::detail::ContiguousIndexToOffset<args, index_t> makeContiguousIndexer()
	{
		return megu::detail::ContiguousIndexToOffset<args, index_t>();
	}

	template<typename T>
	struct PolymorphicIndexToOffset {
		template<int p>
		using calc = std::unique_ptr<detail::IndexToOffset<p, T>>;

		PolymorphicIndexToOffset(const JitFunctionArg& arg)
		{
			int indexed = arg.nargs() - int(arg.is_pointer.count());
			switch (indexed)
			{
			case 1: v = makeIndexer<1, T>(arg); break;
			case 2: v = makeIndexer<2, T>(arg); break;
			case 3: v = makeIndexer<3, T>(arg); break;
			case 4: v = makeIndexer<4, T>(arg); break;
			case 5: v = makeIndexer<5, T>(arg); break;
			case 6: v = makeIndexer<6, T>(arg); break;
			case 7: v = makeIndexer<7, T>(arg); break;
			case 8: v = makeIndexer<8, T>(arg); break;
			case 9: v = makeIndexer<9, T>(arg); break;  
			case 10: v = makeIndexer<10, T>(arg); break;
			default:
				MEGU_ENSURE(false, "At most 10 tensor arguments on jitted kernels")
					break;
			}
		}

		void* ptr()
		{
			return std::visit([](auto& v) { return static_cast<void*>(v.get()); }, v);
		}

	private:
		std::variant<calc<1>, calc<2>, calc<3>,
			calc<4>, calc<5>, calc<6>, calc<7>,
			calc<8>, calc<9>, calc<10>> v;
	};

	static inline int get_vec_size(const dtype_t dtype, void* ptr)
	{
		MEGU_VISIT_DTYPE(
			dtype, scalar_t,
			return memory::get_vec_size<scalar_t>(reinterpret_cast<char*>(ptr));
		)
	}

	ContentArgs::vec_ctx ContentArgs::getVectorizationInfo(JitFunctionArg const& arg)
	{
		dtype_t dtype = arg.arg.compute_type;

		if (arg.is_pointer.count() == arg.arg.data.size()) {//a very special case for very special people
			return { -1,false };
		}
		if (!arg.is_contiguous()) {
			return { -1,false };
		}

		int vecOut = 9999;//whatever lol
		int ptr = 0; 
		for (int i = 0; i < arg.arg.dtype.size(); ++i)
		{
			if (arg.is_scalar[i])
				continue;
			if (dtype != arg.arg.dtype[ptr])
				return { -1,false };
			vecOut = std::min<int>(vecOut, get_vec_size(arg.arg.dtype[ptr], arg.arg.data[ptr])); 
			++ptr;
		}
		if (vecOut > 1) {
			return { vecOut,true };
		}
		return { -1,false };
	}


	using kernel_t = jitify::experimental::KernelInstantiation;
	using program_t = jitify::experimental::Program;


	void launchRawKernel(uintptr_t _kernel , Dim3 const& grid, Dim3 const& block, unsigned int smem,
		uintptr_t stream, void* args[])
    {
        CHECK_CUDA(cuLaunchKernel((CUfunction)_kernel, grid.x, grid.y, grid.z,
            block.x, block.y, block.z, smem, 
            (CUstream)stream, args , 0)); 
    }

	static inline void launch_raw_kernel(CUfunction _kernel, dim3 const& grid, dim3 const& block, unsigned int smem,
		CUstream stream, void* args[])
	{
		CHECK_CUDA(cuLaunchKernel(_kernel, grid.x, grid.y, grid.z,
			block.x, block.y, block.z, smem,
			stream, args, 0));
	}



#define SWITCH_10(n,_)\
       switch(n){\
        case 1: {_(1); } break;\
		case 2: {_(2); } break;\
		case 3: {_(3); } break;\
		case 4: {_(4); } break;\
		case 5: {_(5); } break;\
		case 6: {_(6); } break;\
		case 7: {_(7); } break;\
		case 8: {_(8); } break;\
		case 9: {_(9); } break;\
		case 10: {_(10); } break;\
		default:MEGU_ENSURE(false, "too many arguements only support up to 10;");\
	   }

#define MAKE_INDEXER(obj,ptr,type)\
ptr = std::make_unique<type<N>

	template<int N> 
	static inline void launch_impl(
		JitFunctionArg& arg,
		kernel_t const& kernel,
		ContentArgs const& misc,
		const int64_t  size,
		const dim3& grid,
		const dim3& block,
		cudaStream_t stream)
	{
		SmallVector<void*,10> kargs;
		kargs.reserve(arg.nargs() + arg.scalars.size() + 2);
		auto ptrs = arg.get_data_ptrs<N>();
		kargs.push_back(&ptrs); 
		
		if (misc.vectorization.vectorized) {
			unsigned int s = size;
			kargs.push_back(&s); 
			for (int i = 0; i < arg.scalars.size(); ++i) 
				kargs.push_back(arg.scalars[i].mutable_raw_ptr()); 
			launch_raw_kernel(kernel, grid, block, 0, stream, kargs.data());
		}
		else if (misc.is_contiguous) {
			if (misc.can_use_32bit_indexing) {
				//irrelevant 1b sized type
				auto idxr = makeContiguousIndexer<0, unsigned int>();
				unsigned int s = size; 
				kargs.push_back(&idxr); 
				kargs.push_back(&s); 
				for (int i = 0; i < arg.scalars.size(); ++i)
					kargs.push_back(arg.scalars[i].mutable_raw_ptr());

				launch_raw_kernel(kernel, grid, block, 0, stream, kargs.data());
			}
			else {
				//irrelevant 1b sized type
				auto idxr = makeContiguousIndexer<0, uint64_t>();
				uint64_t s = size;
				kargs.push_back(&idxr); 
				kargs.push_back(&s);
				for (int i = 0; i < arg.scalars.size(); ++i) 
					kargs.push_back(arg.scalars[i].mutable_raw_ptr());
				launch_raw_kernel(kernel, grid, block, 0, stream, kargs.data());
			}
		}
		else {
			if (misc.can_use_32bit_indexing) {
				auto idxr = PolymorphicIndexToOffset<unsigned int>(arg);
				unsigned int s = size;
				kargs.push_back(idxr.ptr());
				kargs.push_back(&s);

				for (int i = 0; i < arg.scalars.size(); ++i) 
					kargs.push_back(arg.scalars[i].mutable_raw_ptr());
				launch_raw_kernel(kernel, grid, block, 0, stream, kargs.data());
			}
			else {
				auto idxr = PolymorphicIndexToOffset<uint64_t>(arg);
				uint64_t s = size; 
				kargs.push_back(idxr.ptr() );
				kargs.push_back(&s);
				for (int i = 0; i < arg.scalars.size(); ++i) 
					kargs.push_back(arg.scalars[i].mutable_raw_ptr());
				launch_raw_kernel(kernel, grid, block, 0, stream, kargs.data());
			}
		}
	}


	static inline void launch(kernel_t const& kernel,JitFunctionArg& arg , ContentArgs misc, cudaStream_t stream)
	{
		
		int const dev = arg.device;
		const int64_t size = arg.len();
		
		int warp;
		cudaDeviceGetAttribute(&warp, cudaDevAttrWarpSize, dev);

		const dim3 block_size(warp*4);
		int const thread_work_size = misc.vectorization.vectorized ? 4 : 1;
		unsigned int const block_work_size = block_size.x * thread_work_size;
		const dim3 grid_size((size + block_work_size - 1) / block_work_size);
		const int n = arg.nargs();

		switch(n){
        case 1: {launch_impl<1>(arg,kernel,misc,size,grid_size,block_size,stream); } break;   
        case 2: {launch_impl<2>(arg,kernel,misc,size,grid_size,block_size,stream); } break;  
        case 3: {launch_impl<3>(arg,kernel,misc,size,grid_size,block_size,stream); } break;  
        case 4: {launch_impl<4>(arg,kernel,misc,size,grid_size,block_size,stream); } break;  
        case 5: {launch_impl<5>(arg,kernel,misc,size,grid_size,block_size,stream); } break;  
        case 6: {launch_impl<6>(arg,kernel,misc,size,grid_size,block_size,stream); } break;  
        case 7: {launch_impl<7>(arg,kernel,misc,size,grid_size,block_size,stream); } break;   
        case 8: {launch_impl<8>(arg,kernel,misc,size,grid_size,block_size,stream); } break;  
        case 9: {launch_impl<9>(arg,kernel,misc,size,grid_size,block_size,stream); } break;  
        case 10: {launch_impl<10>(arg, kernel, misc , size, grid_size, block_size,stream); } break;
        default:MEGU_ENSURE(false,"too many arguements only support up to 10;");
        }

	}


//temporary in linux its /usr/local/smth smth but ideally we will need to find it programmatically or have it as a predefined macro
#define CUDA_PATH "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v11.8\\include"


	std::string getCudaPath() {
		return CUDA_PATH;
	}

	static void invoke(
		const JitFunction& func, 
		JitFunctionArg& arg ,
		bool try_vectorize,
		cudaStream_t stream)
	{
		static std::unordered_map<std::string, kernel_t> jit_cache;
		static std::mutex jit_mutex{};

		int reserve_extra = func.function_body().size() + 6;//5 + nullterm for vecsize string

		auto key = arg.encode(reserve_extra);

		ContentArgs misc(arg,try_vectorize);

		key += char(misc.is_contiguous);
		key += char(misc.can_use_32bit_indexing);
		
		key += char(misc.vectorization.vectorized);
		key += std::to_string(misc.vectorization.vec_size);
		key += func.function_body();

		kernel_t const* kernel;

		auto maybe_cached = jit_cache.find(key); 
		
		if (maybe_cached == jit_cache.end())
		{
			auto source = codegen(func, arg, misc);

			std::string name = (std::string)func.function_name() + "_kernel";
			
			if(misc.vectorization.vectorized)
				name+="_vectorized"+std::to_string(misc.vectorization.vec_size); 

			std::lock_guard<std::mutex> jit_lock(jit_mutex);

			auto item =
				jit_cache.insert({ std::move(key),
				program_t(
				std::move(source), {},
				{ "-std=c++17" ,"--use_fast_math",
			"-I" + getCudaPath() ,}
			).kernel(name).instantiate()
					});

			kernel = &item.first->second;
		}
		else {
			kernel = &maybe_cached->second; 
		}

		launch( *kernel , arg, misc, stream);

	}

	void JitFunction::operator()(JitFunctionArg& arg ,cudaStream_t stream ,bool try_vectorize) const
	{
		if (arg.arg.len() == 0)return;
		invoke(*this, arg,  try_vectorize , stream);
	}


	
	std::string getFile(std::string path)
	{
		std::ifstream stream(path);
		return std::string((
			std::istreambuf_iterator<char>(stream)),
			std::istreambuf_iterator<char>() 
		);
	}

	

	

#undef SWITCH_10
#undef step

}//end megu::cuda::jit