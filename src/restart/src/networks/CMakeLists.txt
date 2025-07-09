set(network_SRC
    anynet.cpp
    cmesh.cpp
    dragonfly.cpp
    fattree.cpp
    flatfly_onchip.cpp
    fly.cpp
    kncube.cpp
    network.cpp
    qtree.cpp
    tree4.cpp
)

set(network_INC
    anynet.hpp
    cmesh.hpp
    dragonfly.hpp
    fattree.hpp
    flatfly_onchip.hpp
    fly.hpp
    kncube.hpp
    network.hpp
    qtree.hpp
    tree4.hpp
)

PREPEND(network_SRC)
PREPEND(network_INC)

set(network_SRC ${network_SRC} PARENT_SCOPE)
set(network_INC ${network_INC} PARENT_SCOPE)