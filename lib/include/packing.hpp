#pragma once

inline uint16_t getDataFP16(const uint8_t* dataBytes, int index)
{
    int cellIndex = index * 2;

    uint16_t lsb = dataBytes[cellIndex];
    uint16_t msb = dataBytes[cellIndex + 1];

    return (msb << 8) | lsb;
}

inline void setDataFP16(uint8_t* dataBytes, int index, uint16_t mask)
{
    int cellIndex = index * 2;

    uint8_t lsb = mask & 0b11111111;
    uint8_t msb = mask >> 8;

    dataBytes[cellIndex]     = lsb;
    dataBytes[cellIndex + 1] = msb;
}
