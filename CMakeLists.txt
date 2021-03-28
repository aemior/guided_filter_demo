cmake_minimum_required(VERSION 3.16)
project(guided_filter)

enable_testing()

# Set up language settings
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED YES)
set(CMAKE_CXX_EXTENSIONS NO)

# Find Halide
find_package(Halide REQUIRED)

# Generator
add_executable(guided_filter.generator guided_filter_generator.cpp)
target_link_libraries(guided_filter.generator PRIVATE Halide::Generator Halide::Tools)

# Filters
add_halide_library(guided_filter FROM guided_filter.generator)
add_halide_library(guided_filter_auto_schedule FROM guided_filter.generator
                   GENERATOR guided_filter
                   AUTOSCHEDULER Halide::Mullapudi2016)


# Main executable
add_executable(guided_filter_process process.cpp)
target_link_libraries(guided_filter_process
                      PRIVATE
                      Halide::ImageIO
                      guided_filter
                      guided_filter_auto_schedule)

enable_testing()
# Test that the app actually works!
set(IMAGE ${CMAKE_CURRENT_LIST_DIR}/imgs/rgb.png)
MESSAGE(${IMAGE})
if (EXISTS ${IMAGE})
    configure_file(${IMAGE} rgb.png COPYONLY)
    add_test(NAME guided_filter_process COMMAND guided_filter_process rgb.png rgb.png 0.0004 out.png)
    set_tests_properties(guided_filter_process PROPERTIES
                         LABELS guided_filter
                         PASS_REGULAR_EXPRESSION "Success!"
                         SKIP_REGULAR_EXPRESSION "\\[SKIP\\]")
endif ()