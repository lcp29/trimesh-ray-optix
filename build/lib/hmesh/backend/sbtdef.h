#pragma once

#include "optix_types.h"
#include <string>

enum SBTType {
    INTERSECTS_ANY,
    count
};

const std::string programNames[] = {
    "intersectsAny"
};

template <typename T>
struct Record {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT)
        char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
