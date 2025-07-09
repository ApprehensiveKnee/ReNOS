
# set sorce files
set (power_SRC
    power_module.cpp
    switch_monitor.cpp
    buffer_monitor.cpp
)

set (power_INC
    power_module.hpp
    switch_monitor.hpp
    buffer_monitor.hpp
)

PREPEND(power_SRC)
PREPEND(power_INC)

set(power_SRC ${power_SRC} PARENT_SCOPE)
set(power_INC ${power_INC} PARENT_SCOPE)