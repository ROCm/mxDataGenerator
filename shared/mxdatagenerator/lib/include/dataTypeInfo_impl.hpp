#pragma once

//
// Generic implementations
//

template <typename DTYPE>
inline bool
    isOne(uint8_t const* scaleBytes, uint8_t const* dataBytes, size_t scaleIndex, size_t dataIndex)
{
    return toDouble<DTYPE>(scaleBytes, dataBytes, scaleIndex, dataIndex) == 1.0;
}

template <typename DTYPE>
inline bool isOnePacked(uint8_t const* scaleBytes,
                        uint8_t const* dataBytes,
                        size_t         scaleIndex,
                        size_t         dataIndex)
{

    return toDoublePacked<DTYPE>(scaleBytes, dataBytes, scaleIndex, dataIndex) == 1.0;
}

template <typename DTYPE>
inline bool isLess(double         val,
                   uint8_t const* scaleBytes,
                   uint8_t const* dataBytes,
                   size_t         scaleIndex,
                   size_t         dataIndex)
{
    return toDouble<DTYPE>(scaleBytes, dataBytes, scaleIndex, dataIndex) < val;
}

template <typename DTYPE>
inline bool isLessPacked(double         val,
                         uint8_t const* scaleBytes,
                         uint8_t const* dataBytes,
                         size_t         scaleIndex,
                         size_t         dataIndex)
{
    return toDoublePacked<DTYPE>(scaleBytes, dataBytes, scaleIndex, dataIndex) < val;
}

template <typename DTYPE>
inline bool isGreater(double         val,
                      uint8_t const* scaleBytes,
                      uint8_t const* dataBytes,
                      size_t         scaleIndex,
                      size_t         dataIndex)
{
    return toDouble<DTYPE>(scaleBytes, dataBytes, scaleIndex, dataIndex) > val;
}

template <typename DTYPE>
inline bool isGreaterPacked(double         val,
                            uint8_t const* scaleBytes,
                            uint8_t const* dataBytes,
                            size_t         scaleIndex,
                            size_t         dataIndex)
{
    return toDoublePacked<DTYPE>(scaleBytes, dataBytes, scaleIndex, dataIndex) > val;
}

template <typename DTYPE>
inline uint getDataSignBits()
{
    return DTYPE::dataInfo.signBits;
}

template <typename DTYPE>
inline uint getDataExponentBits()
{
    return DTYPE::dataInfo.exponentBits;
}

template <typename DTYPE>
inline uint getDataMantissaBits()
{
    return DTYPE::dataInfo.mantissaBits;
}

template <typename DTYPE>
inline uint getDataBias()
{
    return DTYPE::dataInfo.bias;
}

template <typename DTYPE>
inline int getDataUnBiasedEMin()
{
    return DTYPE::dataInfo.unBiasedEMin;
}

template <typename DTYPE>
inline int getDataUnBiasedEMax()
{
    return DTYPE::dataInfo.unBiasedEMax;
}
template <typename DTYPE>
inline int getDataBiasedEMin()
{
    return DTYPE::dataInfo.biasedEMin;
}
template <typename DTYPE>
inline int getDataBiasedEMax()
{
    return DTYPE::dataInfo.biasedEMax;
}
template <typename DTYPE>
inline bool getDataHasInf()
{
    return DTYPE::dataInfo.hasInf;
}
template <typename DTYPE>
inline bool getDataHasNan()
{
    return DTYPE::dataInfo.hasNan;
}
template <typename DTYPE>
inline bool getDataHasZero()
{
    return DTYPE::dataInfo.hasZero;
}

template <typename DTYPE>
inline float getDataMax()
{
    int e = DTYPE::dataInfo.exponentBits, m = DTYPE::dataInfo.mantissaBits;

    if(e == 5 && m == 2) //bf 8
        return 57344;
    else if(e == 4 && m == 3) //fp 4
        return 448;
    else if(e == 3 && m == 2) //bf 6
        return 28;
    else if(e == 2 && m == 3) // fp6
        return 7.5;
    else if(e == 2 && m == 1) // fp4
        return 6;
    else
    { // float values greater than 8 bits
        uint expMask = ((1 << DTYPE::dataInfo.exponentBits) - 2) << DTYPE::dataInfo.mantissaBits;

        uint temp         = 1 << (DTYPE::dataInfo.mantissaBits - 1);
        uint mantissaMask = 1 << DTYPE::dataInfo.mantissaBits | temp | (temp - 1);

        float exp = getExponentValue<uint>(
                        expMask, DTYPE::dataInfo.mantissaBits, DTYPE::dataInfo.exponentBits)
                    - static_cast<int>(DTYPE::dataInfo.bias);

        float mantissa = getMantissaValue<uint>(
            mantissaMask, DTYPE::dataInfo.mantissaBits, DTYPE::dataInfo.exponentBits);

        return std::pow(2, exp) * mantissa;
    }
}

template <typename DTYPE>
inline float getDataMin()
{
    return std::pow(2, 1 - static_cast<int>(DTYPE::dataInfo.bias));
}

template <typename DTYPE>
inline float getDataMaxSubnorm()
{
    uint temp         = 1 << (DTYPE::dataInfo.mantissaBits - 1);
    uint mantissaMask = temp | (temp - 1);

    return getMantissaValue<uint>(
               mantissaMask, DTYPE::dataInfo.mantissaBits, DTYPE::dataInfo.exponentBits)
           * std::pow(2, 1 - static_cast<int>(DTYPE::dataInfo.bias));
}

template <typename DTYPE>
inline float getDataMinSubnorm()
{
    return std::pow(2, -static_cast<int>(DTYPE::dataInfo.mantissaBits))
           * std::pow(2, 1 - static_cast<int>(DTYPE::dataInfo.bias));
}

template <typename DTYPE>
inline uint getScaleSignBits()
{
    return DTYPE::scaleInfo.signBits;
}

template <typename DTYPE>
inline uint getScaleExponentBits()
{
    return DTYPE::scaleInfo.exponentBits;
}

template <typename DTYPE>
inline uint getScaleMantissaBits()
{
    return DTYPE::scaleInfo.mantissaBits;
}

template <typename DTYPE>
inline uint getScaleBias()
{
    return DTYPE::scaleInfo.bias;
}

template <typename DTYPE>
inline int getScaleUnBiasedEMin()
{
    return DTYPE::scaleInfo.unBiasedEMin;
}

template <typename DTYPE>
inline int getScaleUnBiasedEMax()
{
    return DTYPE::scaleInfo.unBiasedEMax;
}

template <typename DTYPE>
inline int getScaleBiasedEMin()
{
    return DTYPE::scaleInfo.biasedEMin;
}

template <typename DTYPE>
inline int getScaleBiasedEMax()
{
    return DTYPE::scaleInfo.biasedEMax;
}

template <typename DTYPE>
inline bool getScaleHasInf()
{
    return DTYPE::scaleInfo.hasInf;
}

template <typename DTYPE>
inline bool getScaleHasNan()
{
    return DTYPE::scaleInfo.hasNan;
}

template <typename DTYPE>
inline bool getScaleHasZero()
{
    return DTYPE::scaleInfo.hasZero;
}

template <typename T, typename DTYPE>
inline T convertToType(float value)
{
    using namespace Constants;

    if(std::abs(value) > getDataMax<DTYPE>())
    {

        float maxVal = getDataMax<DTYPE>();

        union cvt
        {
            float num;
            uint  bRep;
        } t;

        t.num     = maxVal;
        uint bMax = t.bRep;

        t.num      = value;
        T sign     = t.bRep >> 31;
        T exp      = ((bMax >> F32MANTISSABITS) & 0xff) - (127 - getDataBias<DTYPE>());
        T mantissa = bMax >> (F32MANTISSABITS - getDataMantissaBits<DTYPE>());

        uint mPrev = bMax >> (F32MANTISSABITS - getDataMantissaBits<DTYPE>());
        mPrev &= ((1 << getDataMantissaBits<DTYPE>()) - 1);
        mPrev--;

        mPrev <<= (F32MANTISSABITS - getDataMantissaBits<DTYPE>());
        uint prevBit = ((bMax >> 23) << 23) | mPrev;

        t.bRep        = prevBit;
        float prevVal = t.num;
        float diff    = maxVal - prevVal;

        float actualMax = maxVal + (diff / 2);

        if(std::abs(value) < actualMax)
        {
            return sign << ((getDataExponentBits<DTYPE>() + getDataMantissaBits<DTYPE>()))
                   | (exp << getDataMantissaBits<DTYPE>()) | mantissa;
        }
        else
        {
            if(!getDataHasInf<DTYPE>())
            {

                return (1 << (getDataMantissaBits<DTYPE>() + getDataExponentBits<DTYPE>())) - 1;
            }
            else
            {
                exp++;
                return sign << ((getDataExponentBits<DTYPE>() + getDataMantissaBits<DTYPE>()))
                       | (exp << getDataMantissaBits<DTYPE>());
            }
        }
    }
    const int mfmt = F32MANTISSABITS;
    uint32_t  x;
    x = reinterpret_cast<uint32_t&>(value);

    uint32_t head, mantissa;
    int      exponent, bias;
    uint32_t sign;

    head     = x & 0xFF800000;
    mantissa = x & 0x7FFFFF;
    exponent = (head >> 23) & 0xFF;
    sign     = head >> 31;
    bias     = 127;

    uint32_t signed_inf
        = (sign << 7) + (((1 << getDataExponentBits<DTYPE>()) - 1) << getDataMantissaBits<DTYPE>());

    if(x == 0)
    {
        return 0b0;
    }

    const int mini_bias                  = getDataBias<DTYPE>();
    const int mini_denormal_act_exponent = 1 - mini_bias;

    int act_exponent, out_exponent, exponent_diff;

    bool isSubNorm = false;

    if(exponent == 0)
    {
        act_exponent  = exponent - bias + 1;
        exponent_diff = mini_denormal_act_exponent - act_exponent;
        isSubNorm     = true;
    }
    else
    {
        act_exponent = exponent - bias;
        if(act_exponent <= mini_denormal_act_exponent)
        {
            exponent_diff = mini_denormal_act_exponent - act_exponent;
            isSubNorm     = true;
        }
        else
        {
            exponent_diff = 0;
        }
        mantissa += (1 << mfmt);
    }

    bool midpoint = (mantissa & ((1 << (mfmt - getDataMantissaBits<DTYPE>() + exponent_diff)) - 1))
                    == (1 << (mfmt - getDataMantissaBits<DTYPE>() + exponent_diff - 1));

    float minSubNorm = getDataMinSubnorm<DTYPE>() * (sign ? -1 : 1);

    if(isSubNorm && std::abs(value) < std::abs(minSubNorm))
    {
        //closer to 0
        if(std::abs(value) <= std::abs(minSubNorm - value))
            return 0;
        else
            return 1 | (sign << (getDataExponentBits<DTYPE>() + getDataMantissaBits<DTYPE>()));
    }

    if(exponent_diff > 0)
        mantissa >>= exponent_diff;
    else if(exponent_diff == -1)
        mantissa <<= -exponent_diff;
    bool implicit_one = mantissa & (1 << mfmt);
    out_exponent      = (act_exponent + exponent_diff) + mini_bias - (implicit_one ? 0 : 1);

    uint32_t drop_mask = (1 << (mfmt - getDataMantissaBits<DTYPE>())) - 1;
    bool     odd       = mantissa & (1 << (mfmt - getDataMantissaBits<DTYPE>()));
    mantissa += (midpoint ? (odd ? mantissa : mantissa - 1) : mantissa) & drop_mask;

    if(out_exponent == 0)
    {
        if((1 << mfmt) & mantissa)
        {
            out_exponent = 1;
        }
    }
    else
    {
        if((1 << (mfmt + 1)) & mantissa)
        {
            mantissa >>= 1;
            out_exponent++;
        }
    }

    mantissa >>= (mfmt - getDataMantissaBits<DTYPE>());

    const int max_exp = (1 << getDataExponentBits<DTYPE>()) - 1;

    if(out_exponent == 0 && mantissa == 0)
    {
        return 0;
    }

    mantissa &= (1 << getDataMantissaBits<DTYPE>()) - 1;
    return (sign << (getDataExponentBits<DTYPE>() + getDataMantissaBits<DTYPE>()))
           | (out_exponent << getDataMantissaBits<DTYPE>()) | mantissa;
}

//////////////////////////////////////////////////////////////////////////////
// BEGIN OLD DYNAMIC POLYMORPHISM VERSION
//
#if 0
#include "dataTypeInfo.hpp"

inline uint DataTypeInfo::getDataSignBits() const
{
    return dataInfo->signBits;
}

inline uint DataTypeInfo::getDataExponentBits() const
{
    return dataInfo->exponentBits;
}

inline uint DataTypeInfo::getDataMantissaBits() const
{
    return dataInfo->mantissaBits;
}

inline uint DataTypeInfo::getDataBias() const
{
    return dataInfo->bias;
}

inline int DataTypeInfo::getDataUnBiasedEMin() const
{
    return dataInfo->unBiasedEMin;
}

inline int DataTypeInfo::getDataUnBiasedEMax() const
{
    return dataInfo->unBiasedEMax;
}
inline int DataTypeInfo::getDataBiasedEMin() const
{
    return dataInfo->biasedEMin;
}
inline int DataTypeInfo::getDataBiasedEMax() const
{
    return dataInfo->biasedEMax;
}
inline bool DataTypeInfo::getDataHasInf() const
{
    return dataInfo->hasInf;
}
inline bool DataTypeInfo::getDataHasNan() const
{
    return dataInfo->hasNan;
}
inline bool DataTypeInfo::getDataHasZero() const
{
    return dataInfo->hasZero;
}

inline float DataTypeInfo::getDataMax() const
{
    int e = dataInfo->exponentBits, m = dataInfo->mantissaBits;

    if(e == 5 && m == 2) //bf 8
        return 57344;
    else if(e == 4 && m == 3) //fp 4
        return 448;
    else if(e == 3 && m == 2) //bf 6
        return 28;
    else if(e == 2 && m == 3) // fp6
        return 7.5;
    else if(e == 2 && m == 1) // fp4
        return 6;
    else
    { // float values greater than 8 bits
        uint expMask = ((1 << dataInfo->exponentBits) - 2) << dataInfo->mantissaBits;

        uint temp         = 1 << (dataInfo->mantissaBits - 1);
        uint mantissaMask = 1 << dataInfo->mantissaBits | temp | (temp - 1);

        float exp = getExponentValue<uint>(expMask, dataInfo->mantissaBits, dataInfo->exponentBits)
                    - static_cast<int>(dataInfo->bias);

        float mantissa
            = getMantissaValue<uint>(mantissaMask, dataInfo->mantissaBits, dataInfo->exponentBits);

        return std::pow(2, exp) * mantissa;
    }
}

inline float DataTypeInfo::getDataMin() const
{
    return std::pow(2, 1 - static_cast<int>(dataInfo->bias));
}

inline float DataTypeInfo::getDataMaxSubnorm() const
{
    uint temp         = 1 << (dataInfo->mantissaBits - 1);
    uint mantissaMask = temp | (temp - 1);

    return getMantissaValue<uint>(mantissaMask, dataInfo->mantissaBits, dataInfo->exponentBits)
           * std::pow(2, 1 - static_cast<int>(dataInfo->bias));
}

inline float DataTypeInfo::getDataMinSubnorm() const
{
    return std::pow(2, -static_cast<int>(dataInfo->mantissaBits))
           * std::pow(2, 1 - static_cast<int>(dataInfo->bias));
}

inline uint DataTypeInfo::getScaleSignBits() const
{
    return scaleInfo->signBits;
}
inline uint DataTypeInfo::getScaleExponentBits() const
{
    return scaleInfo->exponentBits;
}
inline uint DataTypeInfo::getScaleMantissaBits() const
{
    return scaleInfo->mantissaBits;
}
inline uint DataTypeInfo::getScaleBias() const
{
    return scaleInfo->bias;
}
inline int DataTypeInfo::getScaleUnBiasedEMin() const
{
    return scaleInfo->unBiasedEMin;
}
inline int DataTypeInfo::getScaleUnBiasedEMax() const
{
    return scaleInfo->unBiasedEMax;
}
inline int DataTypeInfo::getScaleBiasedEMin() const
{
    return scaleInfo->biasedEMin;
}
inline int DataTypeInfo::getScaleBiasedEMax() const
{
    return scaleInfo->biasedEMax;
}
inline bool DataTypeInfo::getScaleHasInf() const
{
    return scaleInfo->hasInf;
}
inline bool DataTypeInfo::getScaleHasNan() const
{
    return scaleInfo->hasNan;
}
inline bool DataTypeInfo::getScaleHasZero() const
{
    return scaleInfo->hasZero;
}

inline FloatingPointInfo DataTypeInfo::getDataInfo() const
{
    return *dataInfo;
}
inline FloatingPointInfo DataTypeInfo::getScaleInfo() const
{
    return *scaleInfo;
}

// inline std::shared_ptr<DataTypeInfo> DataTypeInfo::Get(DataType dType)
// {
//     switch(dType)
//     {
//     case BF16:
//         return std::make_shared<DGen::bf16>();
//     case FP16:
//         return std::make_shared<DGen::fp16>();
//     case OCP_E4M3_MXFP8:
//         return std::make_shared<DGen::ocp_e4m3_mxfp8>();
//     case OCP_E5M2_MXFP8:
//         return std::make_shared<DGen::ocp_e5m2_mxfp8>();
//     case OCP_E2M3_MXFP6:
//         return std::make_shared<DGen::ocp_e2m3_mxfp6>();
//     case OCP_E3M2_MXFP6:
//         return std::make_shared<DGen::ocp_e3m2_mxfp6>();
//     case OCP_E2M1_MXFP4:
//         return std::make_shared<DGen::ocp_e2m1_mxfp4>();
//     case F32:
//         return std::make_shared<DGen::f32>();
//     default:
//         throw std::invalid_argument("Invalid DataType value.");
//     }
// }

inline std::ostream& operator<<(std::ostream& os, const DataType& dt)
{
    switch(dt)
    {
    case OCP_E4M3_MXFP8:
        return (os << "OCP_E4M3_MXFP8");
    case OCP_E5M2_MXFP8:
        return (os << "OCP_E5M2_MXFP8");
    case OCP_E2M3_MXFP6:
        return (os << "OCP_E2M3_MXFP6");
    case OCP_E3M2_MXFP6:
        return (os << "OCP_E3M2_MXFP6");
    case OCP_E2M1_MXFP4:
        return (os << "OCP_E2M1_MXFP4");
    case FP16:
        return (os << "FP16");
    case BF16:
        return (os << "BF16");
    case F32:
        return (os << "F32");
    }
    return os;
};
#endif
//
// END OLD DYNAMIC POLYMORPHISM VERSION
//////////////////////////////////////////////////////////////////////////////
