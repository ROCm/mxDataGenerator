
#include <dataTypeInfo.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <time.h>

#include "float_16_data_gen.hpp"

using namespace DGen;
using DT = DGen::fp16;

constexpr double NotANumber = Constants::QNaN;
constexpr double EPSILON
    = 0.0009765625; // 2^-m, for (m)antissa = 10 bits; diff between 1.0 and next fp (1.0009765625)
constexpr double MAXNORM = 65504;
constexpr double MAXSUBNORM
    = constexpr_pow(2, 1 - 15) * constexpr_pow(2, -10) * (constexpr_pow(2, 10) - 1);
constexpr double MINSUBNORM = constexpr_pow(2, 1 - 15) * constexpr_pow(2, -10) * (1);

class fp16_test : public ::testing::Test
{
protected:
    // [E5M10] Element Data, stored as pairs of uint8
    const uint8_t data[20] = {
        // clang-format off
        0b00000000, 0b00000000, // 0
        0b00000000, 0b00111100, // 1
        0b00000001, 0b01111100, // NaN example 1
        0b00000010, 0b01111100, // NaN example 2
        0b00000011, 0b01111100, // NaN example 3
        0b00000000, 0b01111100, // inf
        0b00000000, 0b00000100, // 6.103515625e-05 (min norm)
        0b11111111, 0b01111011, // 65504 (max norm)
        0b00000001, 0b00000000, // 5.960464477539063e-08 (min subnorm)
        0b11111111, 0b00000011  // 6.097555160522461e-05 (max subnorm)
        // clang-format on
    };
    // [E5M10] Negative Elements -- same as data[], but opposite sign
    const uint8_t negativeData[20] = {
        // clang-format off
        0b00000000, 0b10000000, // -0
        0b00000000, 0b10111100, // -1
        0b00000001, 0b11111100, // NaN example 1
        0b00000010, 0b11111100, // NaN example 2
        0b00000011, 0b11111100, // NaN example 3
        0b00000000, 0b11111100, // -inf
        0b00000000, 0b10000100, // -6.103515625e-05 (min norm)
        0b11111111, 0b11111011, // -65504 (max norm)
        0b00000001, 0b10000000, // -5.960464477539063e-08 (min subnorm)
        0b11111111, 0b10000011  // -6.097555160522461e-05 (max subnorm)
        // clang-format on
    };

    int const ALLNUMS = 65536;

    std::vector<float> fp16Values; // Populated during runtime to all decimal values for FP16
    std::vector<float> fp16ValuesSorted; // Sorted variant for binary searching

    void SetUp() override
    {
        fp16Values.resize(ALLNUMS);
        fp16ValuesSorted.resize(ALLNUMS);

        for(int i = 0; i < ALLNUMS; i++)
        {
            float x = convert(i, 10, 5, 15);
            if(!std::isnan(x))
                fp16ValuesSorted[i] = x;
            fp16Values[i] = x;
        }

        std::sort(fp16ValuesSorted.begin(), fp16ValuesSorted.end());
    }

    float getClosestDiff(float num)
    {
        auto lowerIt  = std::lower_bound(fp16ValuesSorted.begin(), fp16ValuesSorted.end(), num);
        auto higherIt = std::upper_bound(fp16ValuesSorted.begin(), fp16ValuesSorted.end(), num);

        lowerIt--;

        return std::min(std::abs(*lowerIt - num),
                        std::min(std::abs(*higherIt - num), std::abs(*(lowerIt + 1) - num)));
    }
};

TEST_F(fp16_test, isOne)
{
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(true, isOne<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOne<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOne<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(fp16_test, isOnePacked)
{
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(true, isOnePacked<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isOnePacked<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(fp16_test, isZero)
{
    EXPECT_EQ(true, isZero<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZero<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(true, isZero<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZero<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(fp16_test, isZeroPacked)
{
    EXPECT_EQ(true, isZeroPacked<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(true, isZeroPacked<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isZeroPacked<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(fp16_test, isNaN)
{
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(true, isNaN<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaN<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaN<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(true, isNaN<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaN<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaN<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaN<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(fp16_test, isNaNPacked)
{
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(true, isNaNPacked<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isNaNPacked<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(fp16_test, isInf)
{
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInf<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInf<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInf<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInf<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(fp16_test, isInfPacked)
{
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 0)); // 0
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 1)); // 1
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInfPacked<DT>(nullptr, data, 0, 5)); // Inf
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 6)); // min normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 7)); // max normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, data, 0, 9)); // max sub-normal

    // ========================================================================================= //

    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 0)); // 0
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 1)); // 1
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 2)); // NaN 1
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 3)); // NaN 2
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 4)); // NaN 3
    EXPECT_EQ(true, isInfPacked<DT>(nullptr, negativeData, 0, 5)); // Inf
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 6)); // min normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 7)); // max normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 8)); // min sub-normal
    EXPECT_EQ(false, isInfPacked<DT>(nullptr, negativeData, 0, 9)); // max sub-normal
}

TEST_F(fp16_test, isSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    for(int i = 0; i < ALLNUMS; i++)
    {
        uint16_t data = static_cast<uint16_t>(i);

        uint8_t lsb = data & 0b11111111;
        uint8_t msb = data >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        uint16_t exp = (data >> getDataMantissaBits<DT>()) & 0b11111;

        double value = toDouble<DT>(temp, temp, 0, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_FALSE(isSubnorm<DT>(temp, 0))
                << std::bitset<16>(i) << ", " << (exp != 0b0) << ", " << std::isnan(value) << ", "
                << std::isinf(value);
        else
            EXPECT_TRUE(isSubnorm<DT>(temp, 0));
    }
}

TEST_F(fp16_test, isSubnormPacked)
{
    uint8_t temp[] = {0b0, 0b0};

    for(int i = 0; i < ALLNUMS; i++)
    {
        uint16_t data = static_cast<uint16_t>(i);

        uint8_t lsb = data & 0b11111111;
        uint8_t msb = data >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        uint16_t exp = (data >> getDataMantissaBits<DT>()) & 0b11111;

        double value = toDoublePacked<DT>(temp, temp, 0, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_FALSE(isSubnormPacked<DT>(temp, 0));
        else
            EXPECT_TRUE(isSubnormPacked<DT>(temp, 0));
    }
}

// true if XN < val (first arg)
TEST_F(fp16_test, isLess)
{
    double values[] = {Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.000005,
                       -0,
                       NotANumber,
                       0,
                       0.000005,
                       0.5,
                       1,
                       5,
                       10,
                       Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(nullptr, data, i, j);
                double negProd = toDouble<DT>(nullptr, negativeData, i, j);

                EXPECT_EQ(prod < values[k], isLess<DT>(values[k], nullptr, data, i, j));
                EXPECT_EQ(negProd < values[k], isLess<DT>(values[k], nullptr, negativeData, i, j));
            }
        }
    }
}

TEST_F(fp16_test, isLessPacked)
{
    double values[] = {Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.000005,
                       -0,
                       NotANumber,
                       0,
                       0.000005,
                       0.5,
                       1,
                       5,
                       10,
                       Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(nullptr, data, i, j);
                double negProd = toDouble<DT>(nullptr, negativeData, i, j);

                EXPECT_EQ(prod < values[k], isLessPacked<DT>(values[k], nullptr, data, i, j));
                EXPECT_EQ(negProd < values[k],
                          isLessPacked<DT>(values[k], nullptr, negativeData, i, j));
            }
        }
    }
}

// true if XN > val (first arg)
TEST_F(fp16_test, isGreater)
{
    double values[] = {Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.000005,
                       -0,
                       NotANumber,
                       0,
                       0.000005,
                       0.5,
                       1,
                       5,
                       10,
                       Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(nullptr, data, i, j);
                double negProd = toDouble<DT>(nullptr, negativeData, i, j);

                EXPECT_EQ(prod > values[k], isGreater<DT>(values[k], nullptr, data, i, j));
                EXPECT_EQ(negProd > values[k],
                          isGreater<DT>(values[k], nullptr, negativeData, i, j));
            }
        }
    }
}

TEST_F(fp16_test, isGreaterPacked)
{
    double values[] = {Constants::NegInf,
                       -10,
                       -5,
                       -1,
                       -0.5,
                       -0.000005,
                       -0,
                       NotANumber,
                       0,
                       0.000005,
                       0.5,
                       1,
                       5,
                       10,
                       Constants::Inf};

    for(int i = 0; i < 6; i++)
    {
        for(int j = 0; j < 10; j++)
        {
            for(int k = 0; k < 15; k++)
            {
                double prod    = toDouble<DT>(nullptr, data, i, j);
                double negProd = toDouble<DT>(nullptr, negativeData, i, j);

                EXPECT_EQ(prod > values[k], isGreaterPacked<DT>(values[k], nullptr, data, i, j));
                EXPECT_EQ(negProd > values[k],
                          isGreaterPacked<DT>(values[k], nullptr, negativeData, i, j));
            }
        }
    }
}

TEST_F(fp16_test, toFloatALLFP16AllValues)
{
    uint8_t temp[2];

    for(int i = 0; i < ALLNUMS; i++)
    {
        uint8_t lsb = static_cast<uint16_t>(i) & 0b11111111;
        uint8_t msb = static_cast<uint16_t>(i) >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        float res      = toFloat<DT>(nullptr, temp, i, 0);
        float expected = fp16Values[i];

        if(std::isnan(expected)) // Don't compare NaN to NaN
            EXPECT_TRUE(std::isnan(res));
        else if(std::isinf(expected))
            EXPECT_TRUE(std::isinf(res));
        else
            EXPECT_NEAR(expected, res, 1e-40) << res << expected;
    }
}

TEST_F(fp16_test, toFloatALLFP16AllValuesPacked)
{
    for(int i = 0; i < ALLNUMS; i++)
    {
        uint8_t temp[2];
        {
            uint8_t lsb = static_cast<uint16_t>(i) & 0b11111111;
            uint8_t msb = static_cast<uint16_t>(i) >> 8;

            *(temp)     = lsb;
            *(temp + 1) = msb;

            float res      = toFloatPacked<DT>(nullptr, temp, i, 0);
            float expected = fp16Values[i];

            if(std::isnan(expected)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(res));
            else if(std::isinf(expected))
                EXPECT_TRUE(std::isinf(res));
            else
            {
                EXPECT_NEAR(expected, res, 1e-40);
            }
        }
    }
}

TEST_F(fp16_test, toDoubleALLFP16AllValues)
{
    for(int i = 0; i < ALLNUMS; i++)
    {
        uint8_t temp[2];
        {
            uint8_t lsb = static_cast<uint16_t>(i) & 0b11111111;
            uint8_t msb = static_cast<uint16_t>(i) >> 8;

            *(temp)     = lsb;
            *(temp + 1) = msb;

            double res      = toDouble<DT>(nullptr, temp, i, 0);
            double expected = fp16Values[i];

            if(std::isnan(expected)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(res));
            else if(std::isinf(expected))
                EXPECT_TRUE(std::isinf(res));

            else
            {
                EXPECT_NEAR(expected, res, 1e-40);
            }
        }
    }
}

TEST_F(fp16_test, toDoubleALLFP16AllValuesPacked)
{
    for(int i = 0; i < ALLNUMS; i++)
    {
        uint8_t temp[2];
        {
            uint8_t lsb = static_cast<uint16_t>(i) & 0b11111111;
            uint8_t msb = static_cast<uint16_t>(i) >> 8;

            *(temp)     = lsb;
            *(temp + 1) = msb;

            double res      = toDoublePacked<DT>(nullptr, temp, i, 0);
            double expected = fp16Values[i];

            if(std::isnan(expected)) // Don't compare NaN to NaN
                EXPECT_TRUE(std::isnan(res));
            else if(std::isinf(expected))
                EXPECT_TRUE(std::isinf(res));
            else
            {
                EXPECT_NEAR(expected, res, 1e-40);
            }
        }
    }
}

TEST_F(fp16_test, setOne)
{
    uint8_t data[] = {0b0, 0b0};
    setOne<DT>(nullptr, data, 0, 0);
    double dElem = toDouble<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(fp16_test, setOnePacked)
{
    uint8_t data[] = {0b0, 0b0};
    setOnePacked<DT>(nullptr, data, 0, 0);
    double dElem = toDoublePacked<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(1.0, dElem);
}

TEST_F(fp16_test, setZero)
{
    uint8_t data[] = {0b0, 0b0};
    setZero<DT>(nullptr, data, 0, 0);
    double dElem = toDouble<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(fp16_test, setZeroPacked)
{
    uint8_t data[] = {0b0, 0b0};
    setZeroPacked<DT>(nullptr, data, 0, 0);
    double dElem = toDoublePacked<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(0.0, dElem);
}

TEST_F(fp16_test, setNaN)
{
    uint8_t data[] = {0b0, 0b0};
    setNaN<DT>(nullptr, data, 0, 0);
    double dElem = toDouble<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(fp16_test, setNaNPacked)
{
    uint8_t data[] = {0b0, 0b0};
    setNaNPacked<DT>(nullptr, data, 0, 0);
    double dElem = toDoublePacked<DT>(nullptr, data, 0, 0);
    EXPECT_EQ(true, std::isnan(dElem));
}

TEST_F(fp16_test, setDataMaxNorm)
{
    uint8_t data[] = {0b0, 0b0};

    setDataMax<DT>(data, 0, false, true); // Leave optional params to normal, pos
    EXPECT_EQ(MAXNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, false, false); // Leave optional params to normal, neg
    EXPECT_EQ(-MAXNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, true, true); // Leave optional params to subnormal, pos
    EXPECT_EQ(MAXSUBNORM, toDouble<DT>(data, data, 0, 0));

    setDataMax<DT>(data, 0, true, false); // Leave optional params to subnormal, neg
    EXPECT_EQ(-MAXSUBNORM, toDouble<DT>(data, data, 0, 0));
}

TEST_F(fp16_test, setDataMaxNormPacked)
{
    uint8_t data[] = {0b0, 0b0};

    setDataMaxPacked<DT>(data, 0);
    EXPECT_EQ(MAXNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, false, false);
    EXPECT_EQ(-MAXNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, true, true);
    EXPECT_EQ(MAXSUBNORM, toDoublePacked<DT>(data, data, 0, 0));

    setDataMaxPacked<DT>(data, 0, true, false);
    EXPECT_EQ(-MAXSUBNORM, toDoublePacked<DT>(data, data, 0, 0));
}

// Saturated tests
TEST_F(fp16_test, satConvertToTypeRounding)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MAXSUBNORM - 1.567;
    uint16_t res  = satConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)))
        << norm << "\n"
        << toDouble<DT>(temp, temp, 0, 0);
}

TEST_F(fp16_test, satConvertToTypeRoundingSmallSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MINSUBNORM + 1e-8;
    uint16_t res  = satConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(MINSUBNORM + 1e-8);
    res  = satConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);
}

TEST_F(fp16_test, satConvertToTypeRoundingLargeSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MAXSUBNORM - 1e-8;
    uint16_t res  = satConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(MAXSUBNORM - 1e-8);
    res  = satConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);
}

TEST_F(fp16_test, satConvertToTypeLarge)
{
    EXPECT_EQ(0b0111101111111111, satConvertToType<DT>(1e60)); // Expect max norm
    EXPECT_EQ(0b1111101111111111, satConvertToType<DT>(-1e60)); // Expect max norm
}

TEST_F(fp16_test, satConvertToTypeMax)
{
    EXPECT_EQ(0b0111101111111111, satConvertToType<DT>(MAXNORM)); // Expect max norm
    EXPECT_EQ(0b1111101111111111, satConvertToType<DT>(-1 * MAXNORM)); // Expect max norm
}

TEST_F(fp16_test, satConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b00000000, satConvertToType<DT>(zero));
}

TEST_F(fp16_test, satConvertToTypeNaN)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = std::numeric_limits<float>::quiet_NaN();
    uint16_t res  = satConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
}

// Generate 1000000 numbers and see if the conversion is good
TEST_F(fp16_test, satConvertToTypeRandom)
{
    double lb = -(MAXNORM + (MAXNORM / 5));
    double ub = MAXNORM + (MAXNORM / 5);

    srandom(time(NULL));

    std::default_random_engine re;
    uint8_t                    temp[] = {0b0, 0b0};

    for(int i = 0; i < 1000000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);

        float rNum = unif(re);

        float closestDiff = getClosestDiff(rNum);

        uint16_t res = satConvertToType<DT>(rNum);

        uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
        uint8_t msb = static_cast<uint16_t>(res) >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        if(std::abs(rNum) < MINSUBNORM)
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
            continue;
        }
        else if(std::isnan(rNum))
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
            continue;
        }

        EXPECT_NEAR(closestDiff, std::abs(rNum - toDouble<DT>(temp, temp, 0, 0)), 1e-40) << rNum;
    }
}

//NON SATURATED TESTS

TEST_F(fp16_test, nonSatConvertToTypeRounding)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MAXSUBNORM - 1.567;
    uint16_t res  = nonSatConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_EQ(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)));
}

TEST_F(fp16_test, nonSatConvertToTypeRoundingSmallSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MINSUBNORM + 1e-8;
    uint16_t res  = nonSatConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(MINSUBNORM + 1e-8);
    res  = nonSatConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = 2.01948e-27;
    res  = nonSatConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toFloat<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(2.01948e-27);
    res  = nonSatConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);

    EXPECT_NEAR(closestDiff, std::abs(norm - toFloat<DT>(temp, temp, 0, 0)), 1e-40);
}

TEST_F(fp16_test, nonSatConvertToTypeRoundingLargeSubnorm)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = MAXSUBNORM - 1e-8;
    uint16_t res  = nonSatConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    double closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);

    norm = -(MAXSUBNORM - 1e-8);
    res  = nonSatConvertToType<DT>(norm);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    closestDiff = getClosestDiff(norm);
    EXPECT_NEAR(closestDiff, std::abs(norm - toDouble<DT>(temp, temp, 0, 0)), 1e-40);
}

TEST_F(fp16_test, nonSatConvertToTypeLarge)
{
    EXPECT_EQ(0b0111110000000000, nonSatConvertToType<DT>(1e60)); // Expect pos inf
    EXPECT_EQ(0b1111110000000000, nonSatConvertToType<DT>(-1e60)); // Expect negative inf

    uint8_t temp[] = {0b0, 0b0};

    float    largeNum = 1e60;
    uint16_t res      = nonSatConvertToType<DT>(largeNum);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(toFloat<DT>(temp, temp, 0, 0) > 0);
    EXPECT_TRUE(std::isinf(toFloat<DT>(temp, temp, 0, 0)));

    largeNum = -1e60;
    res      = nonSatConvertToType<DT>(largeNum);

    lsb = static_cast<uint16_t>(res) & 0b11111111;
    msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;
    EXPECT_TRUE(toFloat<DT>(temp, temp, 0, 0) < 0);
    EXPECT_TRUE(std::isinf(toFloat<DT>(temp, temp, 0, 0)));
}

TEST_F(fp16_test, nonSatConvertToTypeMax)
{
    EXPECT_EQ(0b0111101111111111, nonSatConvertToType<DT>(MAXNORM)); // Expect max norm
    EXPECT_EQ(0b1111101111111111,
              nonSatConvertToType<DT>(-1 * MAXNORM)); // Expect max norm
}

TEST_F(fp16_test, nonSatConvertToTypeZero)
{
    float zero = 0.f;
    EXPECT_EQ(0b00000000, nonSatConvertToType<DT>(zero));
}

TEST_F(fp16_test, nonSatConvertToTypeNaN)
{
    uint8_t temp[] = {0b0, 0b0};

    float    norm = std::numeric_limits<float>::quiet_NaN();
    uint16_t res  = nonSatConvertToType<DT>(norm);

    uint8_t lsb = static_cast<uint16_t>(res) & 0b11111111;
    uint8_t msb = static_cast<uint16_t>(res) >> 8;

    *(temp)     = lsb;
    *(temp + 1) = msb;

    EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
}

// Generate 1000000 numbers and see if the conversion is good
TEST_F(fp16_test, nonSatConvertToTypeRandom)
{
    double lb = -(MAXNORM + (MAXNORM / 5));
    double ub = MAXNORM + (MAXNORM / 5);

    srandom(time(NULL));

    std::default_random_engine re;
    uint8_t                    temp[] = {0b0, 0b0};

    for(int i = 0; i < 1000000; i++)
    {
        std::uniform_real_distribution<float> unif(lb, ub);

        float rNum = unif(re);

        double closestDiff = getClosestDiff(rNum);

        uint16_t res = nonSatConvertToType<DT>(rNum);

        uint8_t lsb = res & 0b11111111;
        uint8_t msb = res >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        if(std::abs(rNum) < MINSUBNORM)
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
            continue;
        }
        else if(std::isnan(rNum))
        {
            EXPECT_TRUE(std::isnan(toDouble<DT>(temp, temp, 0, 0)));
            continue;
        }
        else if(std::abs(rNum) > 65520) // Max Norm plus room for rounding
        {
            EXPECT_TRUE(std::isinf(toDouble<DT>(temp, temp, 0, 0)))
                << "input: " << rNum << "\noutput: " << toDouble<DT>(temp, temp, 0, 0);

            if(rNum < 0)
                EXPECT_TRUE(toDouble<DT>(temp, temp, 0, 0) < 0)
                    << "input: " << rNum << "\noutput: " << toDouble<DT>(temp, temp, 0, 0);
            else
                EXPECT_TRUE(toDouble<DT>(temp, temp, 0, 0) > 0)
                    << "input: " << rNum << "\noutput: " << toDouble<DT>(temp, temp, 0, 0);
            continue;
        }

        EXPECT_NEAR(closestDiff, std::abs(rNum - toDouble<DT>(temp, temp, 0, 0)), 1e-40)
            << rNum << "\n";
    }
}

TEST_F(fp16_test, isSubnormal)
{
    uint8_t temp[] = {0b0, 0b0};

    for(int i = 0; i < ALLNUMS; i++)
    {
        uint16_t data = static_cast<uint16_t>(i);

        uint8_t lsb = data & 0b11111111;
        uint8_t msb = data >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        uint16_t exp = (data >> getDataMantissaBits<DT>()) & 0x1f;

        double value = toDouble<DT>(temp, temp, 0, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_FALSE(isSubnorm<DT>(temp, 0));
        else
            EXPECT_TRUE(isSubnorm<DT>(temp, 0));
    }
}

TEST_F(fp16_test, isSubnormalPacked)
{
    uint8_t temp[] = {0b0, 0b0};

    for(int i = 0; i < ALLNUMS; i++)
    {
        uint16_t data = static_cast<uint16_t>(i);

        uint8_t lsb = data & 0b11111111;
        uint8_t msb = data >> 8;

        *(temp)     = lsb;
        *(temp + 1) = msb;

        uint16_t exp = (data >> getDataMantissaBits<DT>()) & 0x1f;

        double value = toDoublePacked<DT>(temp, temp, 0, 0);

        if(exp != 0b0 || std::isnan(value) || std::isinf(value))
            EXPECT_FALSE(isSubnormPacked<DT>(temp, 0));
        else
            EXPECT_TRUE(isSubnormPacked<DT>(temp, 0));
    }
}

TEST_F(fp16_test, getDataMax)
{
    float mantissa = 1;
    for(int m = 1; m <= 10; m++)
        mantissa += std::pow(2, -m);

    float maxi = std::pow(2, 15) * mantissa; // Multiply max biased exp
    EXPECT_EQ(maxi, getDataMax<DT>());
}

TEST_F(fp16_test, getDataMin)
{
    EXPECT_EQ(std::pow(2, 1 - 15), getDataMin<DT>()); // Min biased exp
}

TEST_F(fp16_test, getDataMaxSubnorm)
{
    float exp      = std::pow(2, 1 - 15); // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * (std::pow(2, mBits) - 1);
    EXPECT_EQ(exp * mantissa, getDataMaxSubnorm<DT>());
}

TEST_F(fp16_test, getDataMinSubnorm)
{
    float exp      = std::pow(2, 1 - 15); // Min biased exp
    float mBits    = getDataMantissaBits<DT>();
    float mantissa = std::pow(2, -mBits) * 1;
    EXPECT_EQ(exp * mantissa, getDataMinSubnorm<DT>());
}

TEST_F(fp16_test, roundToEvenTest)
{

    uint8_t tData[2];

    for(int i = 0; i < (1 << 16); i += 2)
    {
        float    input = (fp16Values[i] + fp16Values[i + 1]) / 2;
        uint16_t temp  = satConvertToType<DT>(input);

        uint8_t lsb = temp & 0b11111111;
        uint8_t msb = temp >> 8;

        *(tData)     = lsb;
        *(tData + 1) = msb;

        float  fOutput = toFloat<DT>(tData, tData, 0, 0);
        double dOutput = toDouble<DT>(tData, tData, 0, 0);

        if(std::isnan(input))
        {
            EXPECT_TRUE(std::isnan(fOutput) && std::isnan(dOutput));
            continue;
        }

        if(std::isinf(input))
        {
            EXPECT_TRUE(std::abs(fOutput) == std::abs(getDataMax<DT>()));
            EXPECT_TRUE(std::abs(dOutput) == std::abs(getDataMax<DT>()));
            continue;
        }

        EXPECT_EQ(fp16Values[i], fOutput);
        EXPECT_EQ(static_cast<double>(fp16Values[i]), dOutput);
    }
}

TEST_F(fp16_test, greaterThanMaxTest)
{

    float max = getDataMax<DT>();
    union cvt
    {
        float num;
        uint  bRep;
    } t;

    t.num     = max;
    uint bMax = t.bRep;

    uint mPrev = bMax >> (Constants::F32MANTISSABITS - getDataMantissaBits<DT>());
    mPrev &= ((1 << getDataMantissaBits<DT>()) - 1);
    mPrev--;

    mPrev <<= (Constants::F32MANTISSABITS - getDataMantissaBits<DT>());
    uint prevBit = ((bMax >> 23) << 23) | mPrev;

    t.bRep        = prevBit;
    float prevVal = t.num;
    float diff    = max - prevVal;

    float actualMax = max + (diff / 2);

    uint8_t tData[2];

    for(float input = max; input <= actualMax + 1000; input += 1.5)
    {
        uint16_t satOutput  = satConvertToType<DT>(input);
        uint16_t nSatOutput = nonSatConvertToType<DT>(input);

        uint8_t msb = satOutput >> 8;
        uint8_t lsb = satOutput & 0xff;

        *tData       = lsb;
        *(tData + 1) = msb;

        EXPECT_EQ(getDataMax<DT>(), toFloat<DT>(tData, tData, 0, 0));

        msb = nSatOutput >> 8;
        lsb = nSatOutput & 0xff;

        *tData        = lsb;
        *(tData + 1)  = msb;
        float nSatVal = toFloat<DT>(tData, tData, 0, 0);

        if(input <= actualMax)
            EXPECT_EQ(getDataMax<DT>(), nSatVal) << "input: " << input << "\noutput: " << nSatVal;
        else
            EXPECT_TRUE(std::isinf(nSatVal));
    }
}
