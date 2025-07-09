
#set the source files
set (arbiters_SRC
    arbiter.cpp
    prio_arb.cpp
    matrix_arb.cpp
    roundrobin_arb.cpp
    tree_arb.cpp
)

#set the header files
set (arbiters_INC
    arbiter.hpp
    prio_arb.hpp
    matrix_arb.hpp
    roundrobin_arb.hpp
)

PREPEND(arbiters_SRC)
PREPEND(arbiters_INC)

set(arbiters_SRC ${arbiters_SRC} PARENT_SCOPE)
set(arbiters_INC ${arbiters_INC} PARENT_SCOPE)