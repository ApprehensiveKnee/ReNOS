
# # Source files list
# set(common_SRC
#     base.cpp
#     booksim_config.cpp
#     buffer_state.cpp
#     buffer.cpp
#     channel.cpp
#     config.cpp
#     misc_utils.cpp
#     outset.cpp
#     packet.cpp
#     packet_reply_info.cpp
#     random_utils.cpp
#     rng_double_wrapper.cpp
#     rng_wrapper.cpp
#     routefunc.cpp
#     stats.cpp
#     vc.cpp
#     trafficmanager.cpp
#     traffic.cpp
#     injection.cpp
#     batchtrafficmanager.cpp
#     booksim_config.cpp
#     simulation_bindings.cpp
# )

# # Include files list
# set(common_INC
#     base.hpp
#     booksim_config.hpp
#     buffer_state.hpp
#     buffer.hpp
#     channel.hpp
#     config.hpp
#     globals.hpp
#     misc_utils.hpp
#     outset.hpp
#     packet.hpp
#     packet_reply_info.hpp
#     params.hpp
#     pipefifo.hpp
#     random_utils.hpp
#     routefunc.hpp
#     stats.hpp
#     vc.hpp
#     trafficmanager.hpp
#     traffic.hpp
#     injection.hpp
#     batchtrafficmanager.hpp
#     booksim_config.hpp
#     logger.hpp
#     simulation_fun.hpp
# )

# PREPEND(common_SRC)
# PREPEND(common_INC)

file(GLOB common_SRC
    *.cpp
)   

file(GLOB common_INC
    *.hpp
)

# pass the variable to parent scope
set(common_SRC ${common_SRC} PARENT_SCOPE)
set(common_INC ${common_INC} PARENT_SCOPE)

