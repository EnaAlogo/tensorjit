#pragma once
#include "tensor.h"
#include <bitset>
#include "jit_function_arg.hpp"


namespace megu::cuda::jit
{
	struct MEGU_API JitArg
	{
		enum class jitType_t :int8_t
		{
			SCALAR,
			TENSOR
		};

		JitArg()
			:t_{ jitType_t::TENSOR }, tensor_{} {};

		JitArg(float arg)
			:t_{ jitType_t::SCALAR }, scalar_{ arg } {};

		JitArg(double arg)
			:t_{ jitType_t::SCALAR }, scalar_{ arg } {};

		JitArg(megu::bfloat16 arg)
			:t_{ jitType_t::SCALAR }, scalar_{ float(arg) } {};
		
		JitArg(megu::half arg)
			:t_{ jitType_t::SCALAR }, scalar_{ float(arg) } {};

		JitArg(int32_t arg)
			:t_{ jitType_t::SCALAR }, scalar_{ arg } {};

		JitArg(int64_t arg)
			:t_{ jitType_t::SCALAR }, scalar_{ arg } {};

		JitArg(int8_t arg)
			:t_{ jitType_t::SCALAR }, scalar_{ arg } {};

		JitArg(bool arg)
			:t_{ jitType_t::SCALAR }, scalar_(arg) {};


		JitArg(std::complex<float> arg)
			:t_{ jitType_t::SCALAR }, scalar_{ static_cast<std::complex<float>>(arg) } {};


		JitArg(const std::complex<double>& arg)
			:t_{ jitType_t::SCALAR }, scalar_{ static_cast<std::complex<double>>(arg) } {};


		JitArg(JitTensor arr)
			:t_{ jitType_t::TENSOR },tensor_(std::move(arr))
		{
			MEGU_ENSURE(tensor_ .realized(), "[JitArg] cannot pass a non realized tensor as an input",
				" pass an empty JitArg instead"); 
	
		}

		JitArg(const Scalar& scalar)
			:t_{ jitType_t::SCALAR }, scalar_{ scalar } {};

		JitArg(JitArg&& other)noexcept { 
			operator=(std::move(other));
		}

		JitArg(JitArg const& other){
			operator=(other);
		}

		JitArg& operator=(JitArg&& other) noexcept {
			if (this != &other) {
				t_ = other.t_; 
				if (other.is_scalar()) {
					scalar_ = std::move(other.scalar_);
				}
				else {
					assert(other.is_tensor() && other.t_ == jitType_t::TENSOR);
					tensor_ = std::move(other.tensor_);
				}
			}
			return *this;
		}

		JitArg& operator=(JitArg const& other){
			if (this != &other) {
				t_ = other.t_;
				if (other.is_scalar()) {
					scalar_ = other.scalar_; 
				}
				else {
					assert(other.is_tensor() && other.t_ == jitType_t::TENSOR);
					tensor_ = other.tensor_.clone();
				}
			}
			return *this;
		}

		~JitArg()
		{
			if (t_ == jitType_t::TENSOR)
				tensor_.~JitTensor();
		}



		[[nodiscard]] bool exists() const
		{
			return t_ == jitType_t::TENSOR ? tensor_.realized() : true;
		}


		[[nodiscard]] bool is_scalar() const { return t_ == jitType_t::SCALAR; };

		[[nodiscard]] bool is_tensor() const { return !is_scalar(); };

		[[nodiscard]] dtype_t dtype() const
		{
			return is_scalar() ? scalar_.dtype() : tensor_.dtype(); 
		}
		[[nodiscard]] Device device() const
		{
			assert(!is_scalar());
			return tensor_.device();
		}
		[[nodiscard]] LongArrayView shape() const
		{
			assert(!is_scalar());
			return tensor_.shape();
		}
		[[nodiscard]] LongArrayView strides() const
		{
			assert(!is_scalar());
			return tensor_.strides();
		}
		
		[[nodiscard]] void* data() const
		{
			assert(!is_scalar());
			return tensor_.data(); 
		}

		[[nodiscard]] int64_t udim(int i)const { 
			return tensor_.udim(i); 
		}
		[[nodiscard]] int64_t stride(int i) const 
		{
			return tensor_.ustride(i); 
		}

		const Scalar& scalar() const { return scalar_; }
		const JitTensor& tensor() const { return tensor_; }

		[[nodiscard]] JitTensor realize();

		void allocate(InjectedAllocator const& alloc , LongArrayView shape, Device dev, dtype_t dt);

		void maybe_broadcast_to(LongArrayView shape);
		
	private:
		union {
			JitTensor tensor_;
			Scalar scalar_;
		};
		jitType_t t_;
	};

	class [[nodiscard]] JitArrayView {
	public:
		using value_type = JitArg;
		using pointer = value_type*;
		using const_pointer = const value_type*;
		using reference = value_type&;
		using const_reference = const value_type&;
		using iterator = pointer;
		using const_iterator = const_pointer;
		using reverse_iterator = std::reverse_iterator<iterator>;
		using const_reverse_iterator = std::reverse_iterator<const_iterator>;
		using size_type = size_t;
		using difference_type = ptrdiff_t;

		/*implicit*/constexpr JitArrayView() = default;

		/*implicit*/constexpr JitArrayView(JitArg& OneElt) : data_{ &OneElt }, size_{ 1 } {}

		/*implicit*/constexpr JitArrayView(JitArg* data, size_t length) : data_{ data }, size_{ length } {}

		constexpr JitArrayView(JitArg* begin, JitArg* end) : data_{ begin }, size_(end - begin) {}

		/*implicit*/constexpr JitArrayView(std::vector<JitArg>& Vec) : data_{ Vec.data() }, size_{ Vec.size() } {}

		template <size_t N>
		/*implicit*/ constexpr JitArrayView(std::array<JitArg, N>& Arr)
			: data_{ Arr.data() }, size_{ N } {}

		template <size_t N>
		/*implicit*/ constexpr JitArrayView(JitArg(&Arr)[N]) : data_{ Arr }, size_{ N } {}

		constexpr JitArrayView(const JitArrayView& other)
			: data_{ other.data_ }, size_{ other.size_ } {}


		constexpr JitArg const* data() const { return data_; }
		constexpr JitArg* data() { return data_; }

		constexpr const_iterator begin() const { return data(); }
		constexpr const_iterator end() const { return data() + size_; }

		constexpr iterator begin() { return data(); }
		constexpr iterator end() { return data() + size_; }

		constexpr const_iterator cbegin() const { return data(); } 
		constexpr const_iterator cend() const { return data() + size_; } 


		constexpr reverse_iterator rbegin() { return reverse_iterator(end()); }
		constexpr reverse_iterator rend() { return reverse_iterator(begin()); }


		constexpr const_reverse_iterator rbegin() const { return const_reverse_iterator(end()); }
		constexpr const_reverse_iterator rend() const { return   const_reverse_iterator(begin()); }

		constexpr JitArg const& front() const {
			assert(!this->empty());
			return data_[0];
		}

		constexpr JitArg const& back() const {
			assert(!this->empty());
			return data_[size_ - 1];
		}
		constexpr JitArg& front() {
			assert(!this->empty());
			return data_[0];
		}

		constexpr JitArg& back() {
			assert(!this->empty());
			return data_[size_ - 1];
		}

		constexpr JitArg const& operator[](size_t Index) const {
			assert(Index < this->size() && "Invalid index!");
			return data_[Index];
		}

		constexpr JitArg& operator[](size_t Index) {
			assert(Index < this->size() && "Invalid index!");
			return data_[Index];
		}

		constexpr size_t size()const
		{
			return size_;
		}

		constexpr bool empty()const
		{
			return size_ == 0;
		}

	private:
		JitArg* data_{ nullptr };
		std::size_t size_{ 0 };
	};
	

	struct MEGU_API JitDeez
	{
		using Self = JitDeez;


		JitDeez(JitArrayView operands ,  ArrayRef<std::string> args , InjectedAllocator cator);


		Self& provideShape(LongArrayView shape)
		{
			this->shape_ = static_cast<ShapeVector>(shape);
			return *this;
		}
		Self& provideDevice(Device dev)
		{
			this->device_ = dev;
			return *this;
		}
		Self& provideDtype(dtype_t dt)
		{
			this->compute_type_ = dt;
			return *this;
		}

		int64_t numel()const;

		Device device()const
		{
			return device_.value();
		}
		bool rank() const
		{
			return shape_.value().size();
		}
		
		bool isInput(int i)const {
			return isConst(i);
		}
		bool isOuput(int i)const { 
			return !isConst(i);
		}
		bool isConst(int i)const {
			return is_const[i];
		}
		bool isPointer(int i)const {
			return is_raw[i];
		}
		bool isScalar(int i)const {
			return operands_[i].is_scalar();
		}

		Self& checkDevices();

		Self& checkSameDtypes();

		Device findDevice() const;

		dtype_t findCommonDtype() const;


		Self& checkShapes();

		ShapeVector getBroadcastShape() const;

		void broadcastAndAllocate();


		bool isSquashableDimension(size_t i, const ShapeVector& shape)const;

		[[nodiscard]] JitTensor realizeOutput(int i)
		{
			return operands_[i].realize();
		}

		
		//builds the ewise arg and the function arg
		[[nodiscard]] JitFunctionArg setArgs(ElementWiseArgument& arg);


		std::tuple<megu::ShapeVector, megu::Axes> squashShape() const;

		//assumes the ewise arg is already built and builds the function arg DONOT PASS A TEMPORARY
		[[nodiscard]] static JitFunctionArg fromEWiseArg(ElementWiseArgument const& arg, ArrayRef<std::string> args,Device dev);


	private:
		JitArrayView operands_; 
		std::bitset<24> is_raw{ 0 };
		std::bitset<24> is_const{ 0 };
		std::optional<Device> device_;
		std::optional<dtype_t> compute_type_;
		/*this is the shape that every input broadcasts to
		* and also every output is required to have this same 
		* shape unless specified to be a pointer(not indexed)
		*/
		std::optional<ShapeVector> shape_;

		InjectedAllocator allocator_;

		static bool is_pointer(std::string_view name);
		static bool is_input(std::string_view name);

		
	};

	
}//end megu::cuda::jit