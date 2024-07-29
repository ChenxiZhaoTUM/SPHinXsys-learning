cmake_minimum_required(VERSION 3.10)
# Extract current folder name and set it as project name
STRING(REGEX REPLACE ".*/(.*)" "\\1" CURRENT_FOLDER ${CMAKE_CURRENT_SOURCE_DIR})
PROJECT("${CURRENT_FOLDER}")

# Specify C++ standard
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set output directories for library and executable
set(LIBRARY_OUTPUT_PATH ${PROJECT_BINARY_DIR}/lib)
set(EXECUTABLE_OUTPUT_PATH "${PROJECT_BINARY_DIR}/bin/")
set(BUILD_INPUT_PATH "${EXECUTABLE_OUTPUT_PATH}/input")
set(BUILD_RELOAD_PATH "${EXECUTABLE_OUTPUT_PATH}/reload")

# Create necessary directories and copy data
file(MAKE_DIRECTORY ${BUILD_INPUT_PATH})
file(COPY ${CMAKE_CURRENT_SOURCE_DIR}/data/ DESTINATION ${BUILD_INPUT_PATH})

# Include directories for headers
include_directories(${CMAKE_SOURCE_DIR}/cppcodec)

# Find all source files in the current directory
aux_source_directory(. DIR_SRCS)

# Add executable target with found source files
add_executable(${PROJECT_NAME} ${DIR_SRCS})

# Include cppcodec headers
include_directories(${CMAKE_SOURCE_DIR}/cppcodec)

# Link any additional libraries (assuming extra_sources_3d is a valid target/library)
# If extra_sources_3d is a directory, you may need to adjust this
target_link_libraries(${PROJECT_NAME} extra_sources_3d)

# Set working directory for the debugger
set_target_properties(${PROJECT_NAME} PROPERTIES VS_DEBUGGER_WORKING_DIRECTORY "${EXECUTABLE_OUTPUT_PATH}")

# Add a test target
add_test(NAME ${PROJECT_NAME}
    COMMAND ${PROJECT_NAME} --state_recording=${TEST_STATE_RECORDING}
    WORKING_DIRECTORY ${EXECUTABLE_OUTPUT_PATH})
set_tests_properties(${PROJECT_NAME} PROPERTIES LABELS "particle relaxation")