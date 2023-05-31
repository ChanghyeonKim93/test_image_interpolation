cmake_minimum_required(VERSION 3.9)
message("CMAKE_SOURCE_DIR: " ${CMAKE_SOURCE_DIR})
project(test_image_interpolation)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++17 -o2")
set(CMAKE_BUILD_TYPE Release)
set(CMAKE_MODULE_PATH 
    ${CMAKE_MODULE_PATH} 
    ${CMAKE_CURRENT_SOURCE_DIR}
)
find_package(Eigen3 REQUIRED)
find_package(OpenCV 4 REQUIRED)

set(INCLUDE_DIRS ${INCLUDE_DIRS} ${CMAKE_CURRENT_BINARY_DIR})
set(INCLUDE_DIRS ${INCLUDE_DIRS} util)
set(SRC ${SRC} 
 util/timer.cpp
 util/image_processing.cpp
#  util/klt_tracker.cpp

 util/converter.cpp
 
 util/feature_extractor.cpp
 util/feature_matcher.cpp
)

include_directories(${INCLUDE_DIRS})

add_executable (test_interp test_interp.cpp ${SRC})
target_link_libraries(test_interp PUBLIC ${OpenCV_LIBS})
target_include_directories(test_interp PUBLIC ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR})
# target_link_libraries(test_interp PUBLIC ${CERES_LIBRARIES})
# target_include_directories(test_interp PUBLIC ${CERES_INCLUDE_DIRS})

# add_executable (test_klt test_klt.cpp ${SRC})
# target_link_libraries(test_klt PUBLIC ${OpenCV_LIBS})
# target_include_directories(test_klt PUBLIC ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR})

add_executable (test_extractor test_extractor.cpp ${SRC})
target_link_libraries(test_extractor PUBLIC ${OpenCV_LIBS})
target_include_directories(test_extractor PUBLIC ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR})


add_executable (test_cache_hit test_cache_hit.cpp ${SRC})
target_link_libraries(test_cache_hit PUBLIC ${OpenCV_LIBS})
target_include_directories(test_cache_hit PUBLIC ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR})

add_executable (test_cache_miss test_cache_miss.cpp ${SRC})
target_link_libraries(test_cache_miss PUBLIC ${OpenCV_LIBS})
target_include_directories(test_cache_miss PUBLIC ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR})


add_executable (test_branch_prediction test_branch_prediction.cpp ${SRC})
target_link_libraries(test_branch_prediction PUBLIC ${OpenCV_LIBS})
target_include_directories(test_branch_prediction PUBLIC ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR})

add_executable (test_branch_misprediction test_branch_misprediction.cpp ${SRC})
target_link_libraries(test_branch_misprediction PUBLIC ${OpenCV_LIBS})
target_include_directories(test_branch_misprediction PUBLIC ${OpenCV_LIBS} ${CMAKE_SOURCE_DIR})