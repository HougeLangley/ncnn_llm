# deps_json.cmake
# Handle nlohmann_json dependency: system package, explicit directory, or FetchContent

find_package(nlohmann_json QUIET)

if(NOT TARGET nlohmann_json AND NOT TARGET nlohmann_json::nlohmann_json)
    if(DEFINED ENV{NLOHMANN_JSON_DIR} AND EXISTS "$ENV{NLOHMANN_JSON_DIR}/include/nlohmann/json.hpp")
        add_library(nlohmann_json INTERFACE)
        target_include_directories(nlohmann_json INTERFACE "$ENV{NLOHMANN_JSON_DIR}/include")
        add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
    else()
        include(FetchContent)
        FetchContent_Declare(
            nlohmann_json
            GIT_REPOSITORY https://github.com/nlohmann/json.git
            GIT_TAG v3.12.0
            GIT_SHALLOW TRUE
        )
        set(JSON_BuildTests OFF CACHE INTERNAL "")
        set(JSON_Install OFF CACHE INTERNAL "")
        FetchContent_MakeAvailable(nlohmann_json)
    endif()
endif()
