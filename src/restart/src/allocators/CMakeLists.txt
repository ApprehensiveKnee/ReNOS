
set( allocators_SRC
    allocator.cpp
    islip.cpp
    loa.cpp
    maxsize.cpp
    pim.cpp
    selalloc.cpp
    separable_input_first.cpp
    separable_output_first.cpp
    separable.cpp
    wavefront.cpp
)

set( allocators_INC
    allocator.hpp
    islip.hpp
    loa.hpp
    maxsize.hpp
    pim.hpp
    selalloc.hpp
    separable_input_first.hpp
    separable_output_first.hpp
    separable.hpp
    wavefront.hpp
)

PREPEND(allocators_SRC)
PREPEND(allocators_INC)

set(allocators_SRC ${allocators_SRC} PARENT_SCOPE)
set(allocators_INC ${allocators_INC} PARENT_SCOPE)
