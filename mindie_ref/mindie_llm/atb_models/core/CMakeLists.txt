file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/*.cpp")

if(NOT USE_W8A16)
    list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/plugin_op/acl_nn_operation.cpp")
    list(REMOVE_ITEM SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/plugin_op/w8a16_operation.cpp")
endif()

add_library(atb_speed_core SHARED ${SOURCE_FILES})
target_link_libraries(atb_speed_core PRIVATE atb)
install(TARGETS atb_speed_core DESTINATION lib)