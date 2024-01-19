file(GLOB_RECURSE SOURCE_FILES "${CMAKE_CURRENT_LIST_DIR}/llama*.cpp ${CMAKE_CURRENT_LIST_DIR}/telechat*.cpp")
add_library(atb_speed_models SHARED ${SOURCE_FILES})
target_link_libraries(atb_speed_models PRIVATE atb atb_speed_core)
install(TARGETS atb_speed_models DESTINATION lib)