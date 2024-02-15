#pragma once
#include <functional>
#include <memory>

namespace megu {

	/*
	* megujit needs to be able to work for many libraries like pytorch tensorflow etc
	* but also its a nice thing for any user to have the choice of providing an allocator
	*/

	using alloc_fn = std::function<void* (size_t)>;
	using dealloc_fn = std::function<void(void*)>;

	using mem_ptr = std::shared_ptr<void>;

	struct InjectedAllocator {
		InjectedAllocator(alloc_fn fn, dealloc_fn dfn)
			:m_alloc(std::move(fn)),
			m_dealloc(std::move(dfn)) {}

		mem_ptr operator()(size_t bytes)const {
			return mem_ptr(m_alloc(bytes), m_dealloc);
		}

	private:
		alloc_fn m_alloc;
		dealloc_fn m_dealloc;

		
	};

};