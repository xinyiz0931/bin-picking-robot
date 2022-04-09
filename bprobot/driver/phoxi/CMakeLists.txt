cmake_minimum_required (VERSION 3.10)

if(POLICY CMP0054)
    cmake_policy(SET CMP0054 NEW)
endif()

project (phoxicontrol)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY .)

find_package(pybind11 REQUIRED)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${phoxicontrol_SOURCE_DIR}) 
pybind11_add_module (phoxicontrol ./phoxicontrol.cpp)

find_package(PhoXi REQUIRED CONFIG PATHS "$ENV{PHOXI_CONTROL_PATH}")

if (UNIX AND NOT APPLE)
    target_link_libraries(phoxicontrol PRIVATE
        ${PHOXI_LIBRARY}
        rt
    )
else (UNIX AND NOT APPLE)
    target_link_libraries(phoxicontrol PRIVATE
        ${PHOXI_LIBRARY} 
    )
endif(UNIX AND NOT APPLE)

target_include_directories (phoxicontrol PRIVATE ${PHOXI_INCLUDE_DIRS})


