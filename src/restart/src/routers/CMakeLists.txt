
set (routers_SRC
    router.cpp
    iq_router.cpp
    event_router.cpp
    chaos_router.cpp
)

set (routers_INC
    router.hpp
    iq_router.hpp
    event_router.hpp
    chaos_router.hpp
)

PREPEND(routers_SRC)
PREPEND(routers_INC)

set(routers_SRC ${routers_SRC} PARENT_SCOPE)
set(routers_INC ${routers_INC} PARENT_SCOPE)