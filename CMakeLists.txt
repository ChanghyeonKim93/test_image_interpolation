cmake_minimum_required(VERSION 3.9)
project(test_image_interpolation)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -o2")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${CMAKE_CURRENT_SOURCE_DIR}
)

# find_package(cmake_modules)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)

set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
set(SRC ${SRC} timer.cpp)

include_directories( ${INCLUDE_DIRS} )

add_executable (test_interp test_interp.cpp ${SRC})
target_link_libraries(test_interp PUBLIC ${OpenCV_LIBS})
target_include_directories(test_interp PUBLIC ${OpenCV_LIBS})
# target_link_libraries(test_interp PUBLIC ${CERES_LIBRARIES})
# target_include_directories(test_interp PUBLIC ${CERES_INCLUDE_DIRS})
