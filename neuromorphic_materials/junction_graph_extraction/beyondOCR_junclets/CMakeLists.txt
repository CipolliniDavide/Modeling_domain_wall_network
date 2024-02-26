cmake_minimum_required(VERSION 3.22)
project(beyondOCR_junclets)

set(CMAKE_CXX_STANDARD 14)

include_directories(.)

add_executable(beyondOCR_junclets
        beyondOCR_junclets.cpp
        dflBinarylib.cpp
        dflBinarylib.h
        dflJuncletslib.cpp
        dflJuncletslib.h
        dflPenWidth.cpp
        dflPenWidth.h
        dflUtils.cpp
        dflUtils.h
        nms.h
        pamImage.cpp
        pamImage.h)
