#pragma once
#include "dataTypeInfo.hpp"
#include "fp6.hpp"

template <>
inline uint8_t oneMask<ocp_e3m2_mxfp6>()
{
    return 0b001100;
}
template <>
inline uint8_t setSignMaskPositive<ocp_e3m2_mxfp6>()
{
    return 0b011111;
}
template <>
inline uint8_t dataMaxPositiveNormalMask<ocp_e3m2_mxfp6>()
{
    return 0b011111;
}
template <>
inline uint8_t dataMaxNegativeNormalMask<ocp_e3m2_mxfp6>()
{
    return 0b111111;
}
template <>
inline uint8_t dataMaxPositiveSubNormalMask<ocp_e3m2_mxfp6>()
{
    return 0b000011;
}
template <>
inline uint8_t dataMaxNegativeSubNormalMask<ocp_e3m2_mxfp6>()
{
    return 0b100011;
}
template <>
inline uint8_t dataSubNormalOneMask<ocp_e3m2_mxfp6>()
{
    return 0b000010;
}
template <>
inline float dataMaxNormalNumber<ocp_e3m2_mxfp6>()
{
    return 28;
}
template <>
inline float dataMinSubNormalNumber<ocp_e3m2_mxfp6>()
{
    return 0.0625;
}
template <>
inline uint8_t positiveZeroMask<ocp_e3m2_mxfp6>()
{
    return 0b000000;
}
template <>
inline uint8_t negativeZeroMask<ocp_e3m2_mxfp6>()
{
    return 0b100000;
}
template <>
inline uint8_t scaleSubNormalOne<ocp_e3m2_mxfp6>()
{
    return Constants::E8M0_3;
}
template <>
inline uint8_t scaleOne<ocp_e3m2_mxfp6>()
{
    return Constants::E8M0_1;
}

template <>
inline bool isNaN<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  size_t         scaleIndex,
                                  size_t         dataIndex)
{
    // no need to check for NAN in dataBytes since theres no NAN representation
    return *(scaleBytes + scaleIndex) == Constants::E8M0_NAN;
}

template <>
inline bool isNaNPacked<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        size_t         scaleIndex,
                                        size_t         dataIndex)
{
    // since the scale is e8m0 and is 8 bits its packed is the same as unpacked
    // as well as there are no NAN for ocp_e3m2_mxfp6
    return isNaN<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex);
}

template <>
inline bool isInf<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                  uint8_t const* dataBytes,
                                  size_t         scaleIndex,
                                  size_t         dataIndex)
{
    // no infinity representation in ocp_e3m2_mxfp6 will always return false
    return false;
}

// no infinity representation in ocp_e3m2_mxfp6 will always return false
template <>
inline bool isInfPacked<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                        uint8_t const* dataBytes,
                                        size_t         scaleIndex,
                                        size_t         dataIndex)
{
    return false;
}

template <>
inline bool isZero<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                   uint8_t const* dataBytes,
                                   size_t         scaleIndex,
                                   size_t         dataIndex)
{
    if(isNaN<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;
    // no need to check for exponent, since e8m0 does not have zero representation
    return (*(dataBytes + dataIndex) & setSignMaskPositive<ocp_e3m2_mxfp6>())
           == positiveZeroMask<ocp_e3m2_mxfp6>();
}

template <>
inline bool isZeroPacked<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                         uint8_t const* dataBytes,
                                         size_t         scaleIndex,
                                         size_t         dataIndex)
{
    if(isNaNPacked<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return false;

    return (getDataFromPackedF6(dataBytes, dataIndex) & setSignMaskPositive<ocp_e3m2_mxfp6>())
           == positiveZeroMask<ocp_e3m2_mxfp6>();
}

template <>
inline bool isSubnorm<ocp_e3m2_mxfp6>(uint8_t const* dataBytes, size_t dataIndex)
{
    // XXX 6 bit
    uint8_t data = *(dataBytes + dataIndex) & 0b00111111;
    return isSubNormal<uint16_t>(
        data, ocp_e3m2_mxfp6::dataInfo.mantissaBits, ocp_e3m2_mxfp6::dataInfo.exponentBits);
}

template <>
inline bool isSubnormPacked<ocp_e3m2_mxfp6>(uint8_t const* dataBytes, size_t dataIndex)
{
    uint8_t data = getDataFromPackedF6(dataBytes, dataIndex);
    return isSubNormal<uint16_t>(
        data, ocp_e3m2_mxfp6::dataInfo.mantissaBits, ocp_e3m2_mxfp6::dataInfo.exponentBits);
}

template <>
inline double toDouble<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                       uint8_t const* dataBytes,
                                       size_t         scaleIndex,
                                       size_t         dataIndex)
{
    if(isNaN<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZero<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0;

    uint8_t data = *(dataBytes + dataIndex) & 0b00111111;

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e3m2_mxfp6::scaleInfo.mantissaBits,
                                             ocp_e3m2_mxfp6::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E3M2_MXFP6_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline double toDoublePacked<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                             uint8_t const* dataBytes,
                                             size_t         scaleIndex,
                                             size_t         dataIndex)
{
    if(isNaNPacked<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<double>::quiet_NaN();

    if(isZeroPacked<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = getDataFromPackedF6(dataBytes, dataIndex);

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e3m2_mxfp6::scaleInfo.mantissaBits,
                                             ocp_e3m2_mxfp6::scaleInfo.exponentBits);

    return convertToDouble<uint8_t, OCP_E3M2_MXFP6_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline float toFloat<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                     uint8_t const* dataBytes,
                                     size_t         scaleIndex,
                                     size_t         dataIndex)
{
    if(isNaN<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZero<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = *(dataBytes + dataIndex) & 0b00111111;

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e3m2_mxfp6::scaleInfo.mantissaBits,
                                             ocp_e3m2_mxfp6::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E3M2_MXFP6_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline float toFloatPacked<ocp_e3m2_mxfp6>(uint8_t const* scaleBytes,
                                           uint8_t const* dataBytes,
                                           size_t         scaleIndex,
                                           size_t         dataIndex)
{
    if(isNaNPacked<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return std::numeric_limits<float>::quiet_NaN();

    if(isZeroPacked<ocp_e3m2_mxfp6>(scaleBytes, dataBytes, scaleIndex, dataIndex))
        return 0.0f;

    uint8_t data = getDataFromPackedF6(dataBytes, dataIndex);

    int scaleExp = getExponentValue<uint8_t>(*(scaleBytes + scaleIndex),
                                             ocp_e3m2_mxfp6::scaleInfo.mantissaBits,
                                             ocp_e3m2_mxfp6::scaleInfo.exponentBits);

    return convertToFloat<uint8_t, OCP_E3M2_MXFP6_DATA, E8M0_SCALE_INFO>(data, scaleExp);
}

template <>
inline void setOne<ocp_e3m2_mxfp6>(
    uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex, bool subNormal)
{
    *(scaleBytes + scaleIndex)
        = subNormal ? scaleSubNormalOne<ocp_e3m2_mxfp6>() : scaleOne<ocp_e3m2_mxfp6>();
    *(dataBytes + dataIndex)
        = subNormal ? dataSubNormalOneMask<ocp_e3m2_mxfp6>() : oneMask<ocp_e3m2_mxfp6>();
}

//set XN = 0, scale X will not be changed
template <>
inline void setZero<ocp_e3m2_mxfp6>(uint8_t* scaleBytes,
                                    uint8_t* dataBytes,
                                    size_t   scaleIndex,
                                    size_t   dataIndex)
{
    *(dataBytes + dataIndex) = positiveZeroMask<ocp_e3m2_mxfp6>();
}

//ocp_e3m2_mxfp6 does not have NAN representation, method will just return
template <>
inline void setNaN<ocp_e3m2_mxfp6>(uint8_t* scaleBytes,
                                   uint8_t* dataBytes,
                                   size_t   scaleIndex,
                                   size_t   dataIndex)
{
    *(scaleBytes + scaleIndex) = Constants::E8M0_NAN;
}

//ocp_e3m2_mxfp6 does not have an infinity representation, method will just return
template <>
inline void setInf<ocp_e3m2_mxfp6>(uint8_t* scaleBytes,
                                   uint8_t* dataBytes,
                                   size_t   scaleIndex,
                                   size_t   dataIndex)
{
    return;
}

template <>
inline void
    setDataMax<ocp_e3m2_mxfp6>(uint8_t* dataBytes, size_t dataIndex, bool subNormal, bool positive)
{
    if(subNormal)
        *(dataBytes + dataIndex) = positive ? dataMaxPositiveSubNormalMask<ocp_e3m2_mxfp6>()
                                            : dataMaxNegativeSubNormalMask<ocp_e3m2_mxfp6>();
    else
        *(dataBytes + dataIndex) = positive ? dataMaxPositiveNormalMask<ocp_e3m2_mxfp6>()
                                            : dataMaxNegativeNormalMask<ocp_e3m2_mxfp6>();
}

template <>
inline void setOnePacked<ocp_e3m2_mxfp6>(
    uint8_t* scaleBytes, uint8_t* dataBytes, size_t scaleIndex, size_t dataIndex, bool subNormal)
{
    *(scaleBytes + scaleIndex) = subNormal ? Constants::E8M0_3 : Constants::E8M0_1;
    uint8_t mask = subNormal ? dataSubNormalOneMask<ocp_e3m2_mxfp6>() : oneMask<ocp_e3m2_mxfp6>();

    setDataPackedF6(dataBytes, dataIndex, mask);
}

//set XN = 0, scale X will not be changed
template <>
inline void setZeroPacked<ocp_e3m2_mxfp6>(uint8_t* scaleBytes,
                                          uint8_t* dataBytes,
                                          size_t   scaleIndex,
                                          size_t   dataIndex)
{
    setDataPackedF6(dataBytes, dataIndex, positiveZeroMask<ocp_e3m2_mxfp6>());
}

template <>
inline void setNaNPacked<ocp_e3m2_mxfp6>(uint8_t* scaleBytes,
                                         uint8_t* dataBytes,
                                         size_t   scaleIndex,
                                         size_t   dataIndex)
{
    *(scaleBytes + scaleIndex) = Constants::E8M0_NAN;
}

//ocp_e3m2_mxfp6 does not have an infinity representation, method will just return
template <>
inline void setInfPacked<ocp_e3m2_mxfp6>(uint8_t* scaleBytes,
                                         uint8_t* dataBytes,
                                         size_t   scaleIndex,
                                         size_t   dataIndex)
{
    return;
}

template <>
inline void setDataMaxPacked<ocp_e3m2_mxfp6>(uint8_t* dataBytes,
                                             size_t   dataIndex,
                                             bool     subNormal,
                                             bool     positive)
{
    uint8_t mask;
    if(subNormal)
        mask = positive ? dataMaxPositiveSubNormalMask<ocp_e3m2_mxfp6>()
                        : dataMaxNegativeSubNormalMask<ocp_e3m2_mxfp6>();
    else
        mask = positive ? dataMaxPositiveNormalMask<ocp_e3m2_mxfp6>()
                        : dataMaxNegativeNormalMask<ocp_e3m2_mxfp6>();

    setDataPackedF6(dataBytes, dataIndex, mask);
}

template <>
inline uint64_t satConvertToType<ocp_e3m2_mxfp6>(float value)
{
    union
    {
        float in;
        uint  bRep;
    } t;
    t.in      = value;
    uint sign = t.bRep >> 31;

    if(std::isnan(value))
    {

        return (sign << 5) | dataMaxPositiveNormalMask<ocp_e3m2_mxfp6>();
    }

    if(std::abs(value) > dataMaxNormalNumber<ocp_e3m2_mxfp6>()) //covers inf case as well
        return value < 0 ? dataMaxNegativeNormalMask<ocp_e3m2_mxfp6>()
                         : dataMaxPositiveNormalMask<ocp_e3m2_mxfp6>();

    uint8_t res = convertToType<uint8_t, ocp_e3m2_mxfp6>(value);

    uint8_t tData[]  = {res};
    uint8_t tScale[] = {Constants::E8M0_1};

    if(std::abs(toFloat<ocp_e3m2_mxfp6>(tScale, tData, 0, 0))
       < dataMinSubNormalNumber<ocp_e3m2_mxfp6>())
        return value < 0 ? negativeZeroMask<ocp_e3m2_mxfp6>() : positiveZeroMask<ocp_e3m2_mxfp6>();

    return res;
}

template <>
inline uint64_t nonSatConvertToType<ocp_e3m2_mxfp6>(float value)
{
    return 0b0;
}
