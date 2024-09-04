#pragma once

namespace DGen
{
    struct OCP_E2M3_MXFP6_DATA
    {
        static constexpr uint signBits     = 1;
        static constexpr uint exponentBits = 2;
        static constexpr uint mantissaBits = 3;
        static constexpr uint bias         = 1;

        static constexpr int unBiasedEMin = 0;
        static constexpr int unBiasedEMax = 2;
        static constexpr int biasedEMin   = 1;
        static constexpr int biasedEMax   = 3;

        static constexpr bool hasInf  = false;
        static constexpr bool hasNan  = false;
        static constexpr bool hasZero = true;
    };

    struct ocp_e2m3_mxfp6
    {
        static constexpr OCP_E2M3_MXFP6_DATA  dataInfo{};
        static constexpr E8M0_SCALE_INFO scaleInfo{};
    };

    #include "ocp_e2m3_mxfp6_impl.hpp"
}
