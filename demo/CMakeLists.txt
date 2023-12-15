find_package(OpenCV REQUIRED)

add_executable(unet_demo unet/unet_demo.cpp)
target_include_directories(unet_demo PUBLIC ${OpenCV_INCLUDE_DIRS})
target_link_libraries(unet_demo PUBLIC graph ${OpenCV_LIBS})