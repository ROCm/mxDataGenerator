#pragma once

#include "fp6.hpp"

namespace DGen
{
    struct OCP_E3M2_MXFP6_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 3;
        static constexpr uint mantissaBits = 2;
        static constexpr uint bias         = 3;

        static constexpr int unBiasedEMin = -1;
        static constexpr int unBiasedEMax = 4;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 7;

        static constexpr bool hasInf  = false;
        static constexpr bool hasNan  = false;
        static constexpr bool hasZero = true;
    };

    struct ocp_e3m2_mxfp6
    {
        static constexpr OCP_E3M2_MXFP6_DATA  dataInfo{};
        static constexpr E8M0_SCALE_INFO scaleInfo{};
    };

#include "ocp_e3m2_mxfp6_impl.hpp"
}
