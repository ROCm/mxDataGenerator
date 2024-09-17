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
inline uint getDataSRShift()
{
    return DTYPE::dataInfo.srShift;
}

template <typename DTYPE>
inline float getDataMax()
{
    int e = DTYPE::dataInfo.exponentBits, m = DTYPE::dataInfo.mantissaBits;

    if(e == 5 && m == 2) // bf8
        return 57344;
    else if(e == 4 && m == 3) //fp8 
        return 448;
    else if(e == 3 && m == 2) //bf6
        return 28;
    else if(e == 2 && m == 3) //fp6
        return 7.5;
    else if(e == 2 && m == 1) //fp4
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

template <typename T, typename DTYPE>
inline T convertToTypeSR(float value, uint seed)
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
            double dmaxVal = static_cast<double>(maxVal);
            double daMax   = static_cast<double>(actualMax);
            double dValue  = static_cast<double>(value);
            double dis     = std::abs(dmaxVal - daMax);
            double dSeed   = static_cast<double>(seed);
            double dProb   = 1.0f - (std::abs(dValue - dmaxVal) / dis); //prob to round down

            double thresh = UINT_MAX * dProb;

            if(!getDataHasInf<DTYPE>() || dSeed <= thresh)
                // return static_cast<T>(satConvertToType(getDataMax<DTYPE>())); //round down time
                return sign == 0 ? DTYPE::dataMaxPositiveNormalMask
                                 : DTYPE::dataMaxNegativeNormalMask;
            else
            {
                exp++;
                return sign << ((getDataExponentBits<DTYPE>()
                                 + getDataMantissaBits<DTYPE>())) // inf
                       | (exp << getDataMantissaBits<DTYPE>());
            }
        }
        else
        {
            if(!getDataHasInf<DTYPE>())
                return (1 << (getDataMantissaBits<DTYPE>() + getDataExponentBits<DTYPE>())) - 1;
            else
            {
                exp++;
                return sign << ((getDataExponentBits<DTYPE>()
                                 + getDataMantissaBits<DTYPE>())) // inf
                       | (exp << getDataMantissaBits<DTYPE>());
            }
        }
    }

    uint32_t f32 = reinterpret_cast<uint32_t&>(value);

    auto f32Man = f32 & 0x7FFFFF;
    auto head   = f32 & 0xFF800000;
    auto f32Exp = (head >> 23) & 0xFF;

    auto signBit = head >> 31;
    auto sign    = signBit << (getDataExponentBits<DTYPE>() + getDataMantissaBits<DTYPE>());

    f32Exp          = (int32_t)f32Exp - 127;
    int32_t exp     = f32Exp;
    auto    man     = f32Man;
    bool    subnorm = false;

    if(value == 0)
        return 0b0;

    if(exp >= getDataUnBiasedEMin<DTYPE>())
    {
        man = f32Man;
    }
    // if the exponent bit is 8, then the subnormal is exactly the same as f32
    else if(exp < getDataUnBiasedEMin<DTYPE>() && getDataExponentBits<DTYPE>() < 8)
    {
        subnorm   = true;
        auto diff = (uint32_t)(getDataUnBiasedEMin<DTYPE>() - exp);
        if(diff >= 32)
        {
            man    = 0;
            f32Man = 0;
        }
        else
        {
            f32Man |= (uint32_t)1 << 23;
            f32Man >>= diff;
        }
        exp = 0;
        man = f32Man;
    }

    uint srShift = getDataSRShift<DTYPE>();

    // For stochastic-rounding we add the aligned random value to the
    // mantissa and then truncate (RTZ).
    man += seed >> srShift;

    // Increment exponent when mantissa overflows due to rounding
    if(man >= (uint32_t)1 << 23)
        ++exp;
    man >>= (23 - getDataMantissaBits<DTYPE>());
    man &= ((1 << getDataMantissaBits<DTYPE>()) - 1);

    auto biasedExp = (uint32_t)exp;
    if(!subnorm)
        biasedExp = (uint32_t)(exp + getDataBias<DTYPE>());
    biasedExp &= ((1 << getDataExponentBits<DTYPE>()) - 1);
    auto val = sign | biasedExp << getDataMantissaBits<DTYPE>() | man;
    return val;
}
