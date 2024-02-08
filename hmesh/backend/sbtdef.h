#pragma once

#include "device_types.h"
#include "optix_types.h"
#include <string>
#include <tuple>

enum SBTType { INTERSECTS_ANY, count };

// if the function contains such program
#define ProgramMask int
// ray generation
#define PRG_RG 1
// intersection
#define PRG_IS 1 << 1
// anyhit
#define PRG_AH 1 << 2
// closest hit
#define PRG_CH 1 << 3
// miss
#define PRG_MS 1 << 4
// direct callable
#define PRG_DC 1 << 5
// continuation callable
#define PRG_CC 1 << 6
// exception
#define PRG_EX 1 << 7

const std::tuple<std::string, ProgramMask> programInfos[] = {
    {"intersectsAny", PRG_RG | PRG_AH | PRG_MS}};

template <typename T> struct SBTRecord {
    __align__(
        OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using SBTRecordEmpty = SBTRecord<void *>;
