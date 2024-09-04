#pragma once
#include "bf16.hpp"
#include "packing.hpp"

//return true iff XN = NAN
template <>
inline bool isNaN<bf16>(uint8_t const* scaleBytes,
                        uint8_t const* dataBytes,
                        size_t         scaleIndex,
                        size_t         dataIndex)
{
    uint16_t data = getDataFP16(dataBytes, dataIndex);

    uint16_t mantissa = data & bf16::dataMantissaMask;
    uint16_t exponent = (data >> bf16::dataInfo.mantissaBits) & 0b11111111;

    return (exponent == 0b11111111 && mantissa != 0b0);
}

// return true iff XN = 0
template <>
inline bool isZero<bf16>(uint8_t const* scaleBytes,
                         uint8_t const* dataBytes,
                         size_t         scaleIndex,
                         size_t         dataIndex)
{

    if(isNaN<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    uint16_t data = getDataFP16(dataBytes, dataIndex);

    return (data & bf16::setSignMask) == bf16::positiveZeroMask;
}

template <>
inline bool isInf<bf16>(uint8_t const* scaleBytes,
                        uint8_t const* dataBytes,
                        size_t         scaleIndex,
                        size_t         dataIndex)
{
    uint16_t data = getDataFP16(dataBytes, dataIndex);
    return (data & bf16::setSignMask) == bf16::dataInfMask;
}

// // return the double value of XN
template <>
inline double toDouble<bf16>(uint8_t const* scaleBytes,
                             uint8_t const* dataBytes,
                             size_t         scaleIndex,
                             size_t         dataIndex)
{

    if(isNaN<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint16_t data = getDataFP16(dataBytes, dataIndex);

    uint16_t signBit = data >> (bf16::dataInfo.exponentBits + bf16::dataInfo.mantissaBits);

    if(isInf<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::infinity() * (signBit ? -1 : 1);

    double d_s = std::pow(-1, static_cast<double>(signBit));

    double d_e;
    if(isSubNormal<uint16_t>(data, bf16::dataInfo.mantissaBits, bf16::dataInfo.exponentBits))
        d_e = std::pow(2, 1 - static_cast<int>(bf16::dataInfo.bias));
    else
        d_e = std::pow(2,
                       getExponentValue<uint16_t>(
                           data, bf16::dataInfo.mantissaBits, bf16::dataInfo.exponentBits)
                           - static_cast<int>(bf16::dataInfo.bias));
    double d_m = getMantissaValue<uint16_t>(
        data, bf16::dataInfo.mantissaBits, bf16::dataInfo.exponentBits);

    return d_s * d_e * d_m;
}

template <>
inline float toFloat<bf16>(uint8_t const* scaleBytes,
                           uint8_t const* dataBytes,
                           size_t         scaleIndex,
                           size_t         dataIndex)
{
    if(isNaN<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint16_t data = getDataFP16(dataBytes, dataIndex);

    uint16_t signBit = data >> (bf16::dataInfo.exponentBits + bf16::dataInfo.mantissaBits);

    if(isInf<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::infinity() * (signBit ? -1 : 1);

    float d_s = std::pow(-1, static_cast<float>(signBit));

    float d_e;
    if(isSubNormal<uint16_t>(data, bf16::dataInfo.mantissaBits, bf16::dataInfo.exponentBits))
        d_e = std::pow(2, 1 - static_cast<int>(bf16::dataInfo.bias));
    else
        d_e = std::pow(2,
                       getExponentValue<uint16_t>(
                           data, bf16::dataInfo.mantissaBits, bf16::dataInfo.exponentBits)
                           - static_cast<int>(bf16::dataInfo.bias));
    float d_m = getMantissaValue<uint16_t>(
        data, bf16::dataInfo.mantissaBits, bf16::dataInfo.exponentBits);

    return d_s * d_e * d_m;
}

// return true iff the number is positive one
template <>
inline bool isOne<bf16>(uint8_t const* scaleBytes,
                        uint8_t const* dataBytes,
                        size_t         scaleIndex,
                        size_t         dataIndex)
{
    return toDouble<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex) == 1.0;
}

//return true iff XN < val
template <>
inline bool isLess<bf16>(double         val,
                         uint8_t const* scaleBytes,
                         uint8_t const* dataBytes,
                         size_t         scaleIndex,
                         size_t         dataIndex)
{
    return toDouble<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex) < val;
}

//return true iff XN > val
template <>
inline bool isGreater<bf16>(double         val,
                            uint8_t const* scaleBytes,
                            uint8_t const* dataBytes,
                            size_t         scaleIndex,
                            size_t         dataIndex)
{
    return toDouble<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex) > val;
}

// return true iff the number is positive one
template <>
inline bool isOnePacked<bf16>(uint8_t const* scaleBytes,
                              uint8_t const* dataBytes,
                              size_t         scaleIndex,
                              size_t         dataIndex)
{

    return isOne<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

// return true iff XN = 0
template <>
inline bool isZeroPacked<bf16>(uint8_t const* scaleBytes,
                               uint8_t const* dataBytes,
                               size_t         scaleIndex,
                               size_t         dataIndex)
{
    return isZero<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//return true iff XN = NAN
template <>
inline bool isNaNPacked<bf16>(uint8_t const* scaleBytes,
                              uint8_t const* dataBytes,
                              size_t         scaleIndex,
                              size_t         dataIndex)
{
    return isNaN<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isInfPacked<bf16>(uint8_t const* scaleBytes,
                              uint8_t const* dataBytes,
                              size_t         scaleIndex,
                              size_t         dataIndex)
{
    return isInf<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//return true iff XN < val
template <>
inline bool isLessPacked<bf16>(double         val,
                               uint8_t const* scaleBytes,
                               uint8_t const* dataBytes,
                               size_t         scaleIndex,
                               size_t         dataIndex)
{
    return isLess<bf16>(val, scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//return true iff XN > val
template <>
inline bool isGreaterPacked<bf16>(double         val,
                                  uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  size_t         scaleIndex,
                                  size_t         dataIndex)
{
    return isGreater<bf16>(val, scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isSubnorm<bf16>(uint8_t const* dataBytes, size_t dataIndex)
{
    uint16_t data = getDataFP16(dataBytes, dataIndex);
    return isSubNormal<uint16_t>(data, bf16::dataInfo.mantissaBits, bf16::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<bf16>(uint8_t const* dataBytes, size_t dataIndex)
{
    return isSubnorm<bf16>(dataBytes, dataIndex);
}

// return the normal double value of XN
template <>
inline double toDoublePacked<bf16>(uint8_t const* scaleBytes,
                                   uint8_t const* dataBytes,
                                   size_t         scaleIndex,
                                   size_t         dataIndex)
{
    return toDouble<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline float toFloatPacked<bf16>(uint8_t const* scaleBytes,
                                 uint8_t const* dataBytes,
                                 size_t         scaleIndex,
                                 size_t         dataIndex)
{
    return toFloat<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//set XN = 1
template <>
inline void setOne<bf16>(
    uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex, bool subNormal)
{
    setDataFP16(dataBytes, dataIndex, bf16::oneMask);
}

//set X = 0
template <>
inline void
    setZero<bf16>(uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex)
{
    setDataFP16(dataBytes, dataIndex, bf16::positiveZeroMask);
}

template <>
inline void
    setNaN<bf16>(uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex)
{
    setDataFP16(dataBytes, dataIndex, bf16::dataNanMask);
}

template <>
inline void
    setInf<bf16>(uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex)
{

    setDataFP16(dataBytes, dataIndex, bf16::dataInfMask);
}

template <>
inline void
    setDataMax<bf16>(uint8_t* dataBytes, size_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        setDataFP16(dataBytes,
                    dataIndex,
                    positive ? bf16::dataMaxPositiveSubNormalMask : bf16::dataMaxNegativeSubNormalMask);

    else
        setDataFP16(
            dataBytes, dataIndex, positive ? bf16::dataMaxPositiveNormalMask : bf16::dataMaxNegativeNormalMask);
}

template <>
inline void setOnePacked<bf16>(
    uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex, bool subNormal)
{
    setOne<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

//set XN = 0, scale X will not be changed
template <>
inline void setZeroPacked<bf16>(uint8_t* scaleBytes,
                                uint8_t* dataBytes,
                                size_t   scaleIndex,
                                size_t   dataIndex)
{
    setZero<bf16>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline void
    setNaNPacked<bf16>(uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex)
{
    setDataFP16(dataBytes, dataIndex, bf16::dataNanMask);
}

template <>
inline void
    setInfPacked<bf16>(uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex)
{
    setDataFP16(dataBytes, dataIndex, bf16::dataInfMask);
}

template <>
inline void
    setDataMaxPacked<bf16>(uint8_t* dataBytes, size_t dataIndex, bool subNormal, bool positive)
{
    setDataMax<bf16>(dataBytes, dataIndex, subNormal, positive);
}

template <>
inline uint64_t satConvertToType<bf16>(float value)
{
    union cvt
    {
        float in;
        uint  bRep;
    } t;
    t.in      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return sign << 15 | bf16::dataNanMask;
    }
    uint16_t res = convertToType<uint16_t, bf16>(value);

    uint8_t tData[] = {0b0, 0b0};

    setDataFP16(tData, 0, res);

    float resVal = toFloat<bf16>(tData, tData, 0, 0);

    if(std::abs(resVal) > bf16::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? bf16::dataMaxNegativeNormalMask : bf16::dataMaxPositiveNormalMask;

    if(std::abs(resVal) < bf16::dataMinSubNormalNumber)
        return value < 0 ? bf16::negativeZeroMask : bf16::positiveZeroMask;

    return res;
}

template <>
inline uint64_t nonSatConvertToType<bf16>(float value)
{
    union cvt
    {
        float in;
        uint  bRep;
    } t;
    t.in      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {
        return sign << 15 | bf16::dataNanMask;
    }

    uint16_t res = convertToType<uint16_t, bf16>(value);

    uint8_t tData[] = {0b0, 0b0};

    setDataFP16(tData, 0, res);

    float resVal = toFloat<bf16>(tData, tData, 0, 0);

    if(std::abs(resVal) > bf16::dataMaxNormalNumber) //covers inf case as well
        return value < 0 ? bf16::dataNegativeInfMask : bf16::dataInfMask;

    if(std::abs(resVal) < bf16::dataMinSubNormalNumber)
        return value < 0 ? bf16::negativeZeroMask : bf16::positiveZeroMask;

    return res;
}
