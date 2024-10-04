#pragma once
#include "dataTypeInfo.hpp"
#include "f32.hpp"

union cvt
{
    float in;
    uint  bRep;
};

inline uint getDataF32(const uint8_t* dataBytes, size_t index)
{
    size_t cellIndex = index * 4;

    uint _1_8   = *(dataBytes + cellIndex);
    uint _9_16  = *(dataBytes + cellIndex + 1);
    uint _17_24 = *(dataBytes + cellIndex + 2);
    uint _25_32 = *(dataBytes + cellIndex + 3);

    return (_25_32 << 24) | (_17_24 << 16) | (_9_16 << 8) | _1_8;
}

inline void setDataF32(uint8_t* dataBytes, size_t index, uint mask)
{
    size_t cellIndex = index * 4;

    uint _1_8   = mask & 0xff;
    uint _9_16  = (mask >> 8) & 0xff;
    uint _17_24 = (mask >> 16) & 0xff;
    uint _25_32 = (mask >> 24) & 0xff;

    *(dataBytes + cellIndex)     = _1_8;
    *(dataBytes + cellIndex + 1) = _9_16;
    *(dataBytes + cellIndex + 2) = _17_24;
    *(dataBytes + cellIndex + 3) = _25_32;
}

// return true iff the number is positive one
template <>
inline bool isOne<f32>(uint8_t const* scaleBytes [[maybe_unused]],
                       uint8_t const* dataBytes,
                       size_t         scaleIndex [[maybe_unused]],
                       size_t         dataIndex)
{
    uint bRep = getDataF32(dataBytes, dataIndex);

    cvt t;
    t.bRep = bRep;

    return t.in == 1.0f;
}

//return true iff XN = NAN
template <>
inline bool isNaN<f32>(uint8_t const* scaleBytes [[maybe_unused]],
                       uint8_t const* dataBytes,
                       size_t         scaleIndex [[maybe_unused]],
                       size_t         dataIndex)
{
    uint bRep = getDataF32(dataBytes, dataIndex);

    cvt t;
    t.bRep = bRep;

    return std::isnan(t.in);
}

// return true iff XN = 0
template <>
inline bool isZero<f32>(uint8_t const* scaleBytes,
                        uint8_t const* dataBytes,
                        size_t         scaleIndex,
                        size_t         dataIndex)
{

    if(isNaN<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    uint bRep = getDataF32(dataBytes, dataIndex);

    cvt t;
    t.bRep = bRep;

    return t.in == 0.0f;
}

template <>
inline bool isInf<f32>(uint8_t const* scaleBytes [[maybe_unused]],
                       uint8_t const* dataBytes,
                       size_t         scaleIndex [[maybe_unused]],
                       size_t         dataIndex)
{
    uint bRep = getDataF32(dataBytes, dataIndex);

    cvt t;
    t.bRep = bRep;

    return std::isinf(t.in);
}

//return true iff XN < val
template <>
inline bool isLess<f32>(double         val,
                        uint8_t const* scaleBytes [[maybe_unused]],
                        uint8_t const* dataBytes,
                        size_t         scaleIndex [[maybe_unused]],
                        size_t         dataIndex)
{
    uint bRep = getDataF32(dataBytes, dataIndex);

    cvt t;
    t.bRep = bRep;

    return static_cast<double>(t.in) < val;
}

//return true iff XN > val
template <>
inline bool isGreater<f32>(double         val,
                           uint8_t const* scaleBytes [[maybe_unused]],
                           uint8_t const* dataBytes,
                           size_t         scaleIndex [[maybe_unused]],
                           size_t         dataIndex)
{
    uint bRep = getDataF32(dataBytes, dataIndex);

    cvt t;
    t.bRep = bRep;

    return static_cast<double>(t.in) > val;
}

// // return the double value of XN
template <>
inline double toDouble<f32>(uint8_t const* scaleBytes [[maybe_unused]],
                            uint8_t const* dataBytes,
                            size_t         scaleIndex [[maybe_unused]],
                            size_t         dataIndex)
{
    uint bRep = getDataF32(dataBytes, dataIndex);

    cvt t;
    t.bRep = bRep;

    return static_cast<double>(t.in);
}

template <>
inline float toFloat<f32>(uint8_t const* scaleBytes [[maybe_unused]],
                          uint8_t const* dataBytes,
                          size_t         scaleIndex [[maybe_unused]],
                          size_t         dataIndex)
{
    uint bRep = getDataF32(dataBytes, dataIndex);

    cvt t;
    t.bRep = bRep;

    return t.in;
}

// return true iff the number is positive one
template <>
inline bool isOnePacked<f32>(uint8_t const* scaleBytes,
                             uint8_t const* dataBytes,
                             size_t         scaleIndex,
                             size_t         dataIndex)
{

    return isOne<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

// return true iff XN = 0
template <>
inline bool isZeroPacked<f32>(uint8_t const* scaleBytes,
                              uint8_t const* dataBytes,
                              size_t         scaleIndex,
                              size_t         dataIndex)
{
    return isZero<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//return true iff XN = NAN
template <>
inline bool isNaNPacked<f32>(uint8_t const* scaleBytes,
                             uint8_t const* dataBytes,
                             size_t         scaleIndex,
                             size_t         dataIndex)
{
    return isNaN<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isInfPacked<f32>(uint8_t const* scaleBytes,
                             uint8_t const* dataBytes,
                             size_t         scaleIndex,
                             size_t         dataIndex)
{
    return isInf<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//return true iff XN < val
template <>
inline bool isLessPacked<f32>(double         val,
                              uint8_t const* scaleBytes,
                              uint8_t const* dataBytes,
                              size_t         scaleIndex,
                              size_t         dataIndex)
{
    return isLess<f32>(val, scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//return true iff XN > val
template <>
inline bool isGreaterPacked<f32>(double         val,
                                 uint8_t const* scaleBytes,
                                 uint8_t const* dataBytes,
                                 size_t         scaleIndex,
                                 size_t         dataIndex)
{
    return isGreater<f32>(val, scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isSubnorm<f32>(uint8_t const* dataBytes, size_t dataIndex)
{
    uint data = getDataF32(dataBytes, dataIndex);
    return isSubNormal<uint>(data, f32::dataInfo.mantissaBits, f32::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<f32>(uint8_t const* dataBytes, size_t dataIndex)
{
    return isSubnorm<f32>(dataBytes, dataIndex);
}

// return the normal double value of XN
template <>
inline double toDoublePacked<f32>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  size_t         scaleIndex,
                                  size_t         dataIndex)
{
    return toDouble<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline float toFloatPacked<f32>(uint8_t const* scaleBytes,
                                uint8_t const* dataBytes,
                                size_t         scaleIndex,
                                size_t         dataIndex)
{
    return toFloat<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//set XN = 1
template <>
inline void setOne<f32>(uint8_t* scaleBytes [[maybe_unused]],
                        uint8_t* dataBytes,
                        size_t   scaleIndex [[maybe_unused]],
                        size_t   dataIndex,
                        bool     subNormal [[maybe_unused]])
{
    setDataF32(dataBytes, dataIndex, f32::oneMask);
}

//set X = 0
template <>
inline void setZero<f32>(uint8_t* scaleBytes [[maybe_unused]],
                         uint8_t* dataBytes,
                         size_t   scaleIndex [[maybe_unused]],
                         size_t   dataIndex)
{
    setDataF32(dataBytes, dataIndex, f32::positiveZeroMask);
}

template <>
inline void setNaN<f32>(uint8_t* scaleBytes [[maybe_unused]],
                        uint8_t* dataBytes,
                        size_t   scaleIndex [[maybe_unused]],
                        size_t   dataIndex)
{
    setDataF32(dataBytes, dataIndex, f32::dataNanMask);
}

template <>
inline void setInf<f32>(uint8_t* scaleBytes [[maybe_unused]],
                        uint8_t* dataBytes,
                        size_t   scaleIndex [[maybe_unused]],
                        size_t   dataIndex)
{

    setDataF32(dataBytes, dataIndex, f32::dataInfMask);
}

template <>
inline void setDataMax<f32>(uint8_t* dataBytes, size_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        setDataF32(dataBytes,
                   dataIndex,
                   positive ? f32::dataMaxPositiveSubNormalMask
                            : f32::dataMaxNegativeSubNormalMask);

    else
        setDataF32(dataBytes,
                   dataIndex,
                   positive ? f32::dataMaxPositiveNormalMask : f32::dataMaxNegativeNormalMask);
}

template <>
inline void setOnePacked<f32>(uint8_t* scaleBytes,
                              uint8_t* dataBytes,
                              size_t   scaleIndex,
                              size_t   dataIndex,
                              bool     subNormal [[maybe_unused]])
{
    setOne<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//set XN = 0, scale X will not be changed
template <>
inline void
    setZeroPacked<f32>(uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex)
{
    setZero<f32>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline void setNaNPacked<f32>(uint8_t* scaleBytes [[maybe_unused]],
                              uint8_t* dataBytes,
                              size_t   scaleIndex [[maybe_unused]],
                              size_t   dataIndex)
{
    setDataF32(dataBytes, dataIndex, f32::dataNanMask);
}

template <>
inline void setInfPacked<f32>(uint8_t* scaleBytes [[maybe_unused]],
                              uint8_t* dataBytes,
                              size_t   scaleIndex [[maybe_unused]],
                              size_t   dataIndex)
{
    setDataF32(dataBytes, dataIndex, f32::dataInfMask);
}

template <>
inline void
    setDataMaxPacked<f32>(uint8_t* dataBytes, size_t dataIndex, bool subNormal, bool positive)
{
    setDataMax<f32>(dataBytes, dataIndex, subNormal, positive);
}

template <>
inline uint64_t satConvertToType<f32>(float value)
{
    cvt t;
    t.in = value;

    return t.bRep;
}

template <>
inline uint64_t nonSatConvertToType<f32>(float value)
{
    cvt t;
    t.in = value;

    return t.bRep;
}

template <>
inline uint64_t satConvertToTypeSR<f32>(float value, uint seed [[maybe_unused]])
{
    cvt t;
    t.in = value;

    return t.bRep;
}

template <>
inline uint64_t nonSatConvertToTypeSR<f32>(float value, uint seed [[maybe_unused]])
{
    cvt t;
    t.in = value;

    return t.bRep;
}
