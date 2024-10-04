#pragma once
#include "dataTypeInfo.hpp"
#include "data_generation_utils.hpp"

#include <limits>

namespace DGen
{
    constexpr int SPRINKLE_BLOCK_MIN = 3;
    constexpr int SPRINKLE_BLOCK_MAX = 15;

    enum DataPattern
    {
        Bounded,
        BoundedAlternatingSign,
        Unbounded,
        Trigonometric,
        Identity,
        Ones,
        Zeros
        // ...
    };

    enum DataScaling
    {
        Mean
        // ...
    };

    struct DataGeneratorOptions
    {
        bool clampToF32  = false;
        bool includeInf  = false;
        bool includeNaN  = false;
        bool forceDenorm = true;

        DataPattern pattern = DataPattern::Bounded;
        double      min     = -1.0;
        double      max     = 1.0;

        DataScaling scaling      = DataScaling::Mean;
        int         blockScaling = 1;
    };

    template <typename DTYPE>
    class DataGenerator
    {
    public:
        using Generator = std::mt19937;
        // generate internal byte buffers/
        DataGenerator& generate(std::vector<int>            size,
                                std::vector<int>            stride,
                                DataGeneratorOptions const& options);

        // get packed data byte buffer.
        std::vector<uint8_t> getDataBytes() const;

        // get packed scale byte buffer.
        std::vector<uint8_t> getScaleBytes() const;

        // set rng seed
        void setSeed(uint seed);

        // get reference double vector.
        std::vector<double> getReferenceDouble() const; // Hopefully won't overflow to NaN/Inf

        // get reference float double vector.
        std::vector<float> getReferenceFloat() const; // Might overflow to NaN/Inf

    private:
        DataGeneratorOptions m_options;

        uint      seed  = 1713573848;
        Generator m_gen = Generator(seed);

        struct BufferDesc
        {
            size_t array_size;
            size_t bit_size;
            size_t byte_size;
            size_t buffer_size;
        };

        BufferDesc m_dataDesc;
        BufferDesc m_scaleDesc;

        std::vector<uint8_t> m_dataBytes;
        std::vector<uint8_t> m_scaleBytes;

        static std::vector<uint8_t> packArray(BufferDesc in_desc, const std::vector<uint8_t>& src);

        void generate_pattern_bounded(const std::vector<int>& size, const std::vector<int>& stride);
        void generate_pattern_bounded_alternating_sign(const std::vector<int>& size,
                                                       const std::vector<int>& stride);
        void generate_pattern_unbounded(const std::vector<int>& size,
                                        const std::vector<int>& stride);
        void generate_pattern_trigonometric(const std::vector<int>& size,
                                            const std::vector<int>& stride);
        void generate_pattern_identity(const std::vector<int>& size,
                                       const std::vector<int>& stride);
        void generate_pattern_ones();

        void dispatch_generate_pattern(const std::vector<int>& size,
                                       const std::vector<int>& stride);

        uint32_t scale_block_mean(const std::vector<uint32_t>& scales,
                                  std::vector<uint64_t>&       data,
                                  int                          block_size);
        uint32_t dispatch_scale_block(const std::vector<uint32_t>& scales,
                                      std::vector<uint64_t>&       data,
                                      int                          block_size);

        void post_sprinkle(const std::vector<int>& size,
                           const std::vector<int>& stride,
                           int32_t                 unbiased_min_exp);
    };

    template <typename DTYPE>
    inline void DataGenerator<DTYPE>::setSeed(uint seed)
    {
        m_gen.seed(seed);
    }

    template <typename DTYPE>
    inline DataGenerator<DTYPE>& DataGenerator<DTYPE>::generate(std::vector<int>            size,
                                                                std::vector<int>            stride,
                                                                DataGeneratorOptions const& options)
    {
        if(size.size() != stride.size())
            throw std::invalid_argument(
                "Invalid dimensions: size and stride vectors must have the same size.");
        if(size.size() == 0)
            throw std::invalid_argument(
                "Invalid dimensions: size and stride vectors must have size greater than 0.");

        // quick return
        if(size.size() == 0)
        {
            m_dataBytes.resize(0);
            m_scaleBytes.resize(0);
            return *this;
        }

        m_options = options;

        // reorder sizes & strides from least to greatest stride
        const auto          n_size = size.size();
        std::vector<size_t> perm(n_size);
        std::vector<int>    sorted_size(n_size);
        std::vector<int>    sorted_stride(n_size);

        std::iota(perm.begin(), perm.end(), 0);
        std::sort(perm.begin(), perm.end(), [&](auto a, auto b) { return stride[a] < stride[b]; });

        for(size_t i = 0; i < n_size; i++)
        {
            sorted_size[i]   = size[perm[i]];
            sorted_stride[i] = stride[perm[i]];

            if(sorted_size[i] <= 0)
                throw std::invalid_argument(
                    "Invalid dimensions: dimension sizes must be greater than 0.");
            if(sorted_stride[i] <= 0)
                throw std::invalid_argument(
                    "Invalid dimensions: dimension strides must be greater than 0.");
        }

        if(sorted_stride[0] != 1)
            throw std::invalid_argument("Invalid dimensions: the smallest stride must be 1.");

        // assume dimension of contiguous elements is a multiple of block size
        if(sorted_size[0] % options.blockScaling != 0)
            throw std::invalid_argument(
                "Invalid block scaling: dimension of contiguous elements must "
                "be a multiple of block size.");

        // find array sizes (unpacked)
        m_dataDesc.array_size = sorted_stride[n_size - 1] * sorted_size[n_size - 1];
        m_dataDesc.bit_size   = getDataSignBits<DTYPE>() + getDataExponentBits<DTYPE>()
                              + getDataMantissaBits<DTYPE>();
        m_dataDesc.byte_size   = (m_dataDesc.bit_size + 7) / 8;
        m_dataDesc.buffer_size = m_dataDesc.byte_size * m_dataDesc.array_size;

        m_scaleDesc = {0, 0, 0, 0};
        if constexpr(isScaled<DTYPE>())
        {
            // block size must be g.t. 0 if type is scaled.
            if(options.blockScaling <= 0)
            {
                throw std::invalid_argument(
                    "Invalid block scaling: block size must be greater than 0 for this data type.");
            }
            m_scaleDesc.array_size = m_dataDesc.array_size / options.blockScaling;
            m_scaleDesc.bit_size   = getScaleSignBits<DTYPE>() + getScaleExponentBits<DTYPE>()
                                   + getScaleMantissaBits<DTYPE>();
            m_scaleDesc.byte_size   = (m_scaleDesc.bit_size + 7) / 8;
            m_scaleDesc.buffer_size = m_scaleDesc.byte_size * m_scaleDesc.array_size;
        }

        // resize byte array
        m_dataBytes.resize(m_dataDesc.buffer_size, 0x00);
        m_scaleBytes.resize(m_scaleDesc.buffer_size, 0x00);

        dispatch_generate_pattern(sorted_size, sorted_stride);

        return *this;
    }

    template <typename DTYPE>
    std::vector<uint8_t> DataGenerator<DTYPE>::getDataBytes() const
    {
        return DataGenerator::packArray(m_dataDesc, m_dataBytes);
    }

    template <typename DTYPE>
    std::vector<uint8_t> DataGenerator<DTYPE>::getScaleBytes() const
    {
        return DataGenerator::packArray(m_scaleDesc, m_scaleBytes);
    }

    template <typename DTYPE>
    std::vector<double> DataGenerator<DTYPE>::getReferenceDouble() const
    {
        std::vector<double> ret(m_dataDesc.array_size);

        const auto block_size
            = (isScaled<DTYPE>() ? m_options.blockScaling : m_dataDesc.array_size);

        for(size_t i = 0; i < m_dataDesc.array_size; i++)
        {
            const auto scale_idx = i / block_size;
            ret[i] = toDouble<DTYPE>(m_scaleBytes.data(), m_dataBytes.data(), scale_idx, i);
        }

        return ret;
    }

    template <typename DTYPE>
    std::vector<float> DataGenerator<DTYPE>::getReferenceFloat() const
    {
        std::vector<float> ret(m_dataDesc.array_size);

        const auto block_size
            = (isScaled<DTYPE>() ? m_options.blockScaling : m_dataDesc.array_size);

        for(size_t i = 0; i < m_dataDesc.array_size; i++)
        {
            const auto scale_idx = i / block_size;
            ret[i] = toFloat<DTYPE>(m_scaleBytes.data(), m_dataBytes.data(), scale_idx, i);
        }

        return ret;
    }

    template <typename DTYPE>
    std::vector<uint8_t> DataGenerator<DTYPE>::packArray(BufferDesc                  src_desc,
                                                         const std::vector<uint8_t>& src)
    {
        if(!(src_desc.bit_size == 0 || src_desc.bit_size == 4 || src_desc.bit_size == 6
             || src_desc.bit_size == 8 || src_desc.bit_size == 16 || src_desc.bit_size == 32
             || src_desc.bit_size == 64))
            throw std::runtime_error("Error: Cannot generate packed array of this data type.");

        if(src_desc.array_size == 0)
            return std::vector<uint8_t>(0);

        if(src_desc.bit_size % 8 == 0)
            return src;

        const auto           packed_buffer_size = (src_desc.bit_size * src_desc.array_size + 7) / 8;
        std::vector<uint8_t> packed_buffer(packed_buffer_size, 0x00);

        switch(src_desc.bit_size)
        {
        case 4:
            for(size_t i = 0; i < src_desc.array_size; i++)
            {
                const auto dst = i / 2;
                if(i % 2 == 0)
                {
                    // save first half, write second half
                    packed_buffer[dst] = (packed_buffer[dst] & 0xf0) | (src[i] & 0x0f);
                }
                else
                {
                    // save second half, write first half
                    packed_buffer[dst] = (packed_buffer[dst] & 0x0f) | (src[i] << 4);
                }
            }
            break;
        case 6:
            for(size_t i = 0; i < src_desc.array_size; i++)
            {
                const auto dst      = (i * 6) / 8;
                const auto offset_i = i % 4;

                switch(offset_i)
                {
                case 0:
                    packed_buffer[dst] = (packed_buffer[dst] & 0xc0) | (src[i] & 0x3f);
                    break;
                case 1:
                    packed_buffer[dst] = (packed_buffer[dst] & 0x3f) | (src[i] << 6);
                    packed_buffer[dst + 1]
                        = (packed_buffer[dst + 1] & 0xf0) | ((src[i] & 0x3c) >> 2);
                    break;
                case 2:
                    packed_buffer[dst] = (packed_buffer[dst] & 0x0f) | (src[i] << 4);
                    packed_buffer[dst + 1]
                        = (packed_buffer[dst + 1] & 0xfc) | ((src[i] & 0x30) >> 4);
                    break;
                case 3:
                    packed_buffer[dst] = (packed_buffer[dst] & 0x03) | (src[i] << 2);
                    break;
                }
            }
            break;
        }

        return packed_buffer;
    }

    template <typename DTYPE>
    inline void DataGenerator<DTYPE>::generate_pattern_bounded(const std::vector<int>& size,
                                                               const std::vector<int>& stride)
    {
        using namespace Constants;

        // checks
        if(m_options.min >= m_options.max)
            throw std::invalid_argument("Invalid bounds: min must be less than max.");

        const auto min = m_options.min;
        const auto max = m_options.max;

        const auto block_size
            = (isScaled<DTYPE>() ? m_options.blockScaling : m_dataDesc.array_size);

        dimension_iterator ditr(size);

        const bool no_neg = (min >= 0);
        const bool no_pos = (max <= 0);

        auto dataBias         = getDataBias<DTYPE>();
        auto dataMantissaBits = getDataMantissaBits<DTYPE>();
        auto dataExponentBits = getDataExponentBits<DTYPE>();
        auto dataUnbiasedEMin = getDataUnBiasedEMin<DTYPE>();
        // setup scale distribution
        int32_t scaleBias   = getScaleBias<DTYPE>();
        int32_t min_scale   = getScaleUnBiasedEMin<DTYPE>();
        int32_t max_scale   = getScaleUnBiasedEMax<DTYPE>();
        int32_t max_pos_exp = max_scale + getDataUnBiasedEMax<DTYPE>();
        int32_t max_neg_exp = max_pos_exp;
        int32_t min_exp     = int32_t(min_scale - int32_t(dataBias));

        if(m_options.clampToF32)
        {
            min_scale   = std::max(F32MINEXP, min_scale);
            max_scale   = std::min(F32MAXEXP, max_scale);
            max_pos_exp = std::min(F32MAXEXP, max_pos_exp);
            max_neg_exp = std::min(F32MAXEXP, max_neg_exp);
            min_exp     = std::max(F32MINEXP, min_exp);
        }

        uint32_t biased_exp_of_max;
        uint32_t biased_exp_of_min;
        uint64_t man_of_min;
        uint64_t man_of_max;

        split_double(max, nullptr, &biased_exp_of_max, &man_of_max);
        split_double(min, nullptr, &biased_exp_of_min, &man_of_min);

        int32_t exp_of_max = int32_t(biased_exp_of_max) - F64BIAS;
        int32_t exp_of_min = int32_t(biased_exp_of_min) - F64BIAS;

        max_neg_exp = std::min(exp_of_min, max_neg_exp);
        max_pos_exp = std::min(exp_of_max, max_pos_exp);

        max_scale = std::min(
            max_scale,
            (min == 0 ? exp_of_max : (max == 0 ? exp_of_min : std::max(exp_of_max, exp_of_min))));

        if(no_neg && min != 0)
        {
            min_scale = std::max(exp_of_min, min_scale);
            min_exp   = std::max(exp_of_min, min_exp);
        }
        else if(no_pos && max != 0)
        {
            min_scale = std::max(exp_of_max, min_scale);
            min_exp   = std::max(exp_of_max, min_exp);
        }

        // maximum requested value cannot be represented as non-zero number in target type
        if(min_exp > std::max(max_pos_exp, max_neg_exp))
        {
            // if zero within bounds -> return zeros
            if(min <= 0 && 0 <= max)
            {
                post_sprinkle(size, stride, min_exp);
                return;
            }
            // else invalid bounds
            else
                throw std::invalid_argument("Invalid bounds: the max magnitude bound cannot be "
                                            "represented as a non-zero "
                                            "number and zero is not include in the bounds.");
        }

        std::uniform_int_distribution<> scale_dist(min_scale, max_scale);

        // other setup
        uint64_t dtype_max_norm;
        uint32_t dtype_max_norm_exp;
        uint64_t dtype_max_norm_man;

        setDataMax<DTYPE>(reinterpret_cast<uint8_t*>(&dtype_max_norm), 0, false, true);
        split_dynamic(dtype_max_norm,
                      dataExponentBits,
                      dataMantissaBits,
                      nullptr,
                      &dtype_max_norm_exp,
                      &dtype_max_norm_man);

        int32_t dtype_max_norm_biased_exp = dtype_max_norm_exp - dataBias;

        int32_t ub_block_scale = 0;

        for(auto itr = ditr.begin(); itr != ditr.end(); ++itr)
        {
            //
            // compute indices
            //
            int data_i  = get_strided_idx(*itr, stride);
            int scale_i = (data_i / block_size);
            int block_i = data_i % block_size;

            //
            // generate random block
            //
            if constexpr(isScaled<DTYPE>())
            {
                if(block_i == 0)
                {
                    ub_block_scale      = scale_dist(m_gen);
                    int32_t block_scale = ub_block_scale + scaleBias;
                    std::memcpy(&m_scaleBytes[scale_i], &block_scale, m_scaleDesc.byte_size);
                }
            }

            uint64_t result;

            // generate sign
            bool sign;
            if(no_neg || int32_t(ub_block_scale - int32_t(dataBias)) > max_neg_exp)
                sign = 0;
            else if(no_pos || int32_t(ub_block_scale - int32_t(dataBias)) > max_pos_exp)
                sign = 1;
            else
            {
                std::bernoulli_distribution sign_dist(0.5);
                sign = sign_dist(m_gen);
            }

            // generate exponent
            int32_t ub_exp;
            int32_t exp;

            int32_t exp_dist_max = std::min((sign ? max_neg_exp : max_pos_exp) - ub_block_scale,
                                            dtype_max_norm_biased_exp);
            int32_t exp_dist_min = std::max(min_exp - ub_block_scale, -int32_t(dataBias));
            std::uniform_int_distribution<> exp_dist(exp_dist_min, exp_dist_max);

            ub_exp = exp_dist(m_gen);
            exp    = ub_exp + int32_t(dataBias);

            // generate mantissa
            uint64_t mantissa = 0;

            uint64_t mantissa_dist_min = 0;
            uint64_t mantissa_dist_max = (1 << dataMantissaBits) - 1;

            if(ub_exp + ub_block_scale == min_exp)
            {
                if(no_neg && min != 0)
                {
                    mantissa_dist_min = man_of_min >> (52 - dataMantissaBits);
                }
                else if(no_pos && max != 0)
                {
                    mantissa_dist_min = man_of_max >> (52 - dataMantissaBits);
                }
            }

            // if subnorm and subnorm exceeds min exponent
            if(exp == 0 && min_exp > dataUnbiasedEMin + ub_block_scale - int32_t(dataMantissaBits))
            {
                uint64_t temp     = 1ULL << int32_t(int32_t(dataMantissaBits) + min_exp
                                                - int32_t(dataUnbiasedEMin) - ub_block_scale);
                mantissa_dist_min = std::max(mantissa_dist_min, temp);
            }

            // if subnormal exp and zero not included
            if(exp == 0 && (min > 0 || max < 0))
            {
                mantissa_dist_min = std::max(mantissa_dist_min, uint64_t(1));
            }

            if(ub_exp + ub_block_scale == (sign ? max_neg_exp : max_pos_exp))
            {
                mantissa_dist_max = (sign ? man_of_min : man_of_max) >> (52 - dataMantissaBits);
            }

            if(ub_exp == dtype_max_norm_biased_exp)
            {
                mantissa_dist_max = std::min(mantissa_dist_max, dtype_max_norm_man);
            }

            std::uniform_int_distribution<> man_dist(mantissa_dist_min, mantissa_dist_max);
            mantissa = man_dist(m_gen);

            // assemble
            result = uint64_t(sign) << (dataExponentBits + dataMantissaBits);
            result |= uint64_t(exp) << dataMantissaBits;
            result |= mantissa;

            std::memcpy(&m_dataBytes[data_i * m_dataDesc.byte_size], &result, m_dataDesc.byte_size);
        }

        post_sprinkle(size, stride, min_exp);
    }

    template <typename DTYPE>
    void DataGenerator<DTYPE>::generate_pattern_bounded_alternating_sign(
        const std::vector<int>& size, const std::vector<int>& stride)
    {
        using namespace Constants;

        const auto max = std::abs(m_options.max);

        const auto block_size
            = (isScaled<DTYPE>() ? m_options.blockScaling : m_dataDesc.array_size);

        dimension_iterator ditr(size);

        // setup scale distribution
        int32_t max_scale = getScaleUnBiasedEMax<DTYPE>();
        int32_t max_exp   = getDataUnBiasedEMax<DTYPE>() + max_scale;

        if(m_options.clampToF32)
        {
            // min_scale = std::max(F32MINEXP, min_scale);
            max_scale = std::min(F32MAXEXP, max_scale);
            max_exp   = std::min(F32MAXEXP, max_exp);
            // min_exp   = std::max(F32MINEXP, min_exp);
        }

        uint32_t biased_exp_of_max;
        uint64_t man_of_max;
        split_double(max, nullptr, &biased_exp_of_max, &man_of_max);

        int32_t exp_of_max = biased_exp_of_max - F64BIAS;

        max_exp   = std::min(exp_of_max, max_exp);
        max_scale = std::min(exp_of_max, max_scale);

        std::uniform_int_distribution<> scale_dist(getScaleUnBiasedEMin<DTYPE>(), max_scale);

        // other setup
        uint64_t dtype_max_norm;
        uint32_t dtype_max_norm_exp;
        uint64_t dtype_max_norm_man;

        setDataMax<DTYPE>(reinterpret_cast<uint8_t*>(&dtype_max_norm), 0, false, true);
        split_dynamic(dtype_max_norm,
                      getDataExponentBits<DTYPE>(),
                      getDataMantissaBits<DTYPE>(),
                      nullptr,
                      &dtype_max_norm_exp,
                      &dtype_max_norm_man);

        int32_t dtype_max_norm_biased_exp = dtype_max_norm_exp - getDataBias<DTYPE>();

        int32_t ub_block_scale = 0;
        int32_t block_scale    = 0;

        for(auto itr = ditr.begin(); itr != ditr.end(); ++itr)
        {
            //
            // compute indices
            //
            int data_i  = get_strided_idx(*itr, stride);
            int scale_i = (data_i / block_size);
            int block_i = data_i % block_size;

            //
            // generate random block
            //
            if(isScaled<DTYPE>() && block_i == 0)
            {
                ub_block_scale = scale_dist(m_gen);
                block_scale    = ub_block_scale + getScaleBias<DTYPE>();
                std::memcpy(&m_scaleBytes[scale_i], &block_scale, m_scaleDesc.byte_size);
            }

            uint64_t result;

            // generate sign
            bool sign = data_i % 2;

            // generate exponent
            int32_t ub_exp;
            int32_t exp;

            int32_t exp_dist_max = std::min(max_exp - ub_block_scale, dtype_max_norm_biased_exp);
            int32_t exp_dist_min = -int32_t(getDataBias<DTYPE>());

            // set value to 0
            if(exp_dist_max < exp_dist_min)
            {
                continue;
            }

            std::uniform_int_distribution<> exp_dist(exp_dist_min, exp_dist_max);
            ub_exp = exp_dist(m_gen);
            exp    = ub_exp + getDataBias<DTYPE>();

            // generate mantissa
            uint64_t mantissa = 0;

            uint64_t mantissa_dist_min = 0;
            uint64_t mantissa_dist_max = (1 << getDataMantissaBits<DTYPE>()) - 1;

            if(ub_exp + ub_block_scale == max_exp)
            {
                mantissa_dist_max = man_of_max >> (52 - getDataMantissaBits<DTYPE>());
            }

            if(ub_exp == dtype_max_norm_biased_exp)
            {
                mantissa_dist_max = std::min(mantissa_dist_max, dtype_max_norm_man);
            }

            std::uniform_int_distribution<> man_dist(mantissa_dist_min, mantissa_dist_max);
            mantissa = man_dist(m_gen);

            // assemble
            result = uint64_t(sign)
                     << (getDataExponentBits<DTYPE>() + getDataMantissaBits<DTYPE>());
            result |= uint64_t(exp) << getDataMantissaBits<DTYPE>();
            result |= mantissa;

            std::memcpy(&m_dataBytes[data_i * m_dataDesc.byte_size], &result, m_dataDesc.byte_size);
        }

        post_sprinkle(size, stride, isScaled<DTYPE>() ? -F64BIAS : -F32BIAS);
    }

    template <typename DTYPE>
    void DataGenerator<DTYPE>::generate_pattern_unbounded(const std::vector<int>& size,
                                                          const std::vector<int>& stride)
    {
        using namespace Constants;

        const auto block_size
            = (isScaled<DTYPE>() ? m_options.blockScaling : m_dataDesc.array_size);

        dimension_iterator ditr(size);

        auto dataBias          = getDataBias<DTYPE>();
        auto dataMantissaBits  = getDataMantissaBits<DTYPE>();
        auto dataExponentBits  = getDataExponentBits<DTYPE>();
        auto dataUnbiasedEMin  = getDataUnBiasedEMin<DTYPE>();
        auto scaleBias         = getScaleBias<DTYPE>();
        auto scaleBiasedEMax   = getScaleBiasedEMax<DTYPE>();
        auto scaleBiasedEMin   = getScaleBiasedEMin<DTYPE>();
        auto scaleUnbiasedEMax = getScaleUnBiasedEMax<DTYPE>();
        auto scaleUnbiasedEMin = getScaleUnBiasedEMin<DTYPE>();

        int32_t subnorm_min_exp = dataUnbiasedEMin - dataMantissaBits;

        const uint64_t max = (uint64_t(1) << m_dataDesc.bit_size) - 1;

        std::uniform_int_distribution<uint64_t> data_dist(0, max);

        int32_t max_exp = std::numeric_limits<int32_t>::min();
        int32_t min_exp = std::numeric_limits<int32_t>::max();

        for(auto itr = ditr.begin(); itr != ditr.end(); ++itr)
        {
            //
            // compute indices
            //
            size_t data_i  = get_strided_idx(*itr, stride);
            size_t scale_i = (data_i / block_size);
            size_t block_i = data_i % block_size;

            //
            // generate random block
            //
            uint64_t d;
            do
            {
                d = data_dist(m_gen);
            } while(
                (!m_options.includeNaN
                 && isNaN<DTYPE>(
                     reinterpret_cast<uint8_t*>(&scaleBias), reinterpret_cast<uint8_t*>(&d), 0, 0))
                || (!m_options.includeInf
                    && isInf<DTYPE>(reinterpret_cast<uint8_t*>(&scaleBias),
                                    reinterpret_cast<uint8_t*>(&d),
                                    0,
                                    0)));

            const int32_t exp = getExponentValue(d, dataMantissaBits, dataExponentBits) - dataBias;

            max_exp = std::max(max_exp, exp);

            if(exp != -int32_t(dataBias))
            {
                min_exp = std::min(min_exp, exp);
            }
            else
            {
                min_exp = std::min(min_exp, subnorm_min_exp);
            }

            std::memcpy(&m_dataBytes[data_i * m_dataDesc.byte_size], &d, m_dataDesc.byte_size);

            //
            // Generate scale
            //
            if(isScaled<DTYPE>() && block_i == block_size - 1)
            {
                int32_t scaleMax = scaleBiasedEMax;
                int32_t scaleMin = scaleBiasedEMin;

                if(m_options.clampToF32)
                {
                    scaleMax = std::min(F32MAXEXP - max_exp, scaleUnbiasedEMax) + scaleBias;
                    scaleMin = std::max(F32MINEXP - min_exp, scaleUnbiasedEMin) + scaleBias;
                }

                std::uniform_int_distribution<> scale_dist(scaleMin, scaleMax);

                const auto s = scale_dist(m_gen);
                std::memcpy(&m_scaleBytes[scale_i], &s, m_scaleDesc.byte_size);

                // reset per block values
                max_exp = std::numeric_limits<int32_t>::min();
                min_exp = std::numeric_limits<int32_t>::max();
            }
        }

        post_sprinkle(size, stride, isScaled<DTYPE>() ? -F64BIAS : -F32BIAS);
    }

    template <typename DTYPE>
    void DataGenerator<DTYPE>::generate_pattern_trigonometric(const std::vector<int>& size,
                                                              const std::vector<int>& stride)
    {
        // setup
        auto       dataBias          = getDataBias<DTYPE>();
        auto       dataMantissaBits  = getDataMantissaBits<DTYPE>();
        auto       dataExponentBits  = getDataExponentBits<DTYPE>();
        auto       dataUnbiasedEMin  = getDataUnBiasedEMin<DTYPE>();
        auto       scaleBias         = getScaleBias<DTYPE>();
        auto       scaleUnbiasedEMin = getScaleUnBiasedEMin<DTYPE>();
        const auto min_exp = scaleUnbiasedEMin + dataUnbiasedEMin - dataMantissaBits /*subnormal*/;

        const auto block_size
            = (isScaled<DTYPE>() ? m_options.blockScaling : m_dataDesc.array_size);

        dimension_iterator ditr(size);

        std::vector<uint64_t> temp_data((isScaled<DTYPE>() ? block_size : 0), 0);
        std::vector<uint32_t> temp_scale((isScaled<DTYPE>() ? block_size : 0), 0);

        std::uniform_real_distribution<> angle_dist(0.0, 2.0 * M_PI);

        for(auto itr = ditr.begin(); itr != ditr.end(); ++itr)
        {
            //
            // compute indices
            //
            size_t data_i  = (size_t)get_strided_idx(*itr, stride);
            size_t scale_i = (data_i / block_size);
            size_t block_i = data_i % block_size;

            //
            // generate random block
            //

            // generate angle between 0 and 2pi
            const auto angle = angle_dist(m_gen);

            // generate value between -1 and 1
            auto value = std::cos(angle);

            value = m_options.clampToF32 ? static_cast<float>(value) : value;

            // split input
            uint8_t  value_sign;
            uint32_t value_biased_exp;
            uint64_t value_mantissa;
            split_double(value, &value_sign, &value_biased_exp, &value_mantissa);

            const int32_t value_unbiased_exp = value_biased_exp - Constants::F64BIAS;

            // value magnitude must be less than 1;
            if(value_unbiased_exp > 0)
                throw std::runtime_error("Internal Error");

            uint32_t scale = scaleBias;

            // set sign
            uint64_t result
                = value_sign ? (uint64_t(1) << (dataExponentBits + dataMantissaBits)) : 0;

            // if subnormal -> return 0

            // if normal but less than representable range with scale -> return 0

            // within representable range
            if(value_unbiased_exp >= int32_t(min_exp))
            {
                // subnormal
                if(value_unbiased_exp < scaleUnbiasedEMin + dataUnbiasedEMin)
                {
                    // set biased scale
                    scale = scaleUnbiasedEMin + scaleBias;

                    // set exponent -> biased exp = 0

                    // set mantissa (round to zero)
                    // get bits from (data_info.unBiasedEMin - 1) to (data_info.unBiasedEMin - data_info.mantissaBits)
                    //  - get bits that fit in mantissa
                    //  - set implied 1
                    //  - shift remaining exponent
                    uint64_t res_mantissa = value_mantissa >> (53 - dataMantissaBits);
                    res_mantissa |= uint64_t(1) << (dataMantissaBits - 1);
                    res_mantissa >>= scaleUnbiasedEMin + dataUnbiasedEMin - value_unbiased_exp - 1;

                    result |= res_mantissa;
                }
                else
                {
                    // set biased scale and adjust exponent s.t. exponent = 0 if possible
                    int32_t scaled_exp;
                    if(value_unbiased_exp < scaleUnbiasedEMin)
                    {
                        scale      = scaleUnbiasedEMin + scaleBias;
                        scaled_exp = value_unbiased_exp - scaleUnbiasedEMin;
                    }
                    else if(value_unbiased_exp < 0)
                    {
                        scale      = value_unbiased_exp + scaleBias;
                        scaled_exp = 0;
                    }
                    else
                    {
                        scale      = scaleBias;
                        scaled_exp = value_unbiased_exp;
                    }

                    // set exponent
                    const uint32_t result_exp = scaled_exp + dataBias;
                    result |= (result_exp & ((uint64_t(1) << dataExponentBits) - 1))
                              << dataMantissaBits;

                    // set mantissa (round to zero)
                    result |= value_mantissa >> (52 - dataMantissaBits);
                }
            }

            if constexpr(isScaled<DTYPE>())
            {
                temp_data[block_i]  = result;
                temp_scale[block_i] = scale;
            }
            else
            {
                std::memcpy(
                    &m_dataBytes[data_i * m_dataDesc.byte_size], &result, m_dataDesc.byte_size);
            }

            //
            // Compute block scale and adjust data values
            //
            if constexpr(isScaled<DTYPE>())
            {
                if(block_i == block_size - 1)
                {
                    const uint32_t block_scale
                        = dispatch_scale_block(temp_scale, temp_data, block_size);

                    // Write to array
                    for(size_t i = 0; i < block_size; i++)
                    {
                        std::memcpy(&m_dataBytes[(scale_i * block_size + i) * m_dataDesc.byte_size],
                                    &temp_data[i],
                                    m_dataDesc.byte_size);
                    }
                    std::memcpy(&m_scaleBytes[scale_i], &block_scale, m_scaleDesc.byte_size);
                }
            }
        }

        post_sprinkle(size, stride, isScaled<DTYPE>() ? -Constants::F64BIAS : -Constants::F32BIAS);
    }

    template <typename DTYPE>
    void DataGenerator<DTYPE>::generate_pattern_identity(const std::vector<int>& size,
                                                         const std::vector<int>& stride)
    {
        if(size.size() != 2)
            throw std::invalid_argument(
                "Invalid dimensions: Identity data pattern is valid only for two dimensions.");

        const auto rank = std::min(size[0], size[1]);

        std::vector<uint8_t> d_one(m_dataDesc.byte_size, 0x00);
        std::vector<uint8_t> s_one(m_scaleDesc.byte_size, 0x00);

        setOne<DTYPE>(s_one.data(), d_one.data(), 0, 0, m_options.forceDenorm);

        for(int i = 0; i < rank; i++)
        {
            const auto diag_index = i * stride[0] + i * stride[1];
            const auto data_index = diag_index * m_dataDesc.byte_size;
            const auto scale_index
                = isScaled<DTYPE>() ? (diag_index / m_options.blockScaling) * m_scaleDesc.byte_size
                                    : 0;

            std::memcpy(&m_dataBytes[data_index], &d_one[0], m_dataDesc.byte_size);

            if(isScaled<DTYPE>())
            {
                std::memcpy(&m_scaleBytes[scale_index], &s_one[0], m_scaleDesc.byte_size);
            }
        }
    }

    template <typename DTYPE>
    inline void DataGenerator<DTYPE>::generate_pattern_ones()
    {
        std::vector<uint8_t> d_one(m_dataDesc.byte_size, 0x00);
        std::vector<uint8_t> s_one(m_scaleDesc.byte_size, 0x00);

        setOne<DTYPE>(s_one.data(), d_one.data(), 0, 0, m_options.forceDenorm);

        for(size_t i = 0; i < m_dataDesc.array_size; i++)
        {
            std::memcpy(&m_dataBytes[i * m_dataDesc.byte_size], &d_one[0], m_dataDesc.byte_size);
        }

        for(size_t i = 0; i < m_scaleDesc.array_size; i++)
        {
            std::memcpy(&m_scaleBytes[i * m_scaleDesc.byte_size], &s_one[0], m_scaleDesc.byte_size);
        }
    }

    template <typename DTYPE>
    inline void DataGenerator<DTYPE>::dispatch_generate_pattern(const std::vector<int>& size,
                                                                const std::vector<int>& stride)
    {
        switch(m_options.pattern)
        {
        case Bounded:
            return generate_pattern_bounded(size, stride);
        case BoundedAlternatingSign:
            return generate_pattern_bounded_alternating_sign(size, stride);
        case Unbounded:
            return generate_pattern_unbounded(size, stride);
        case Trigonometric:
            return generate_pattern_trigonometric(size, stride);
        case Identity:
            return generate_pattern_identity(size, stride);
        case Ones:
            return generate_pattern_ones();
        case Zeros:
            return;
        }
    }

    template <typename DTYPE>
    inline uint32_t DataGenerator<DTYPE>::scale_block_mean(const std::vector<uint32_t>& scales,
                                                           std::vector<uint64_t>&       data,
                                                           int                          block_size)
    {
        auto dataBias         = getDataBias<DTYPE>();
        auto dataMantissaBits = getDataMantissaBits<DTYPE>();
        auto dataExponentBits = getDataExponentBits<DTYPE>();
        auto dataUnbiasedEMin = getDataUnBiasedEMin<DTYPE>();
        auto dataHasInf       = getDataHasInf<DTYPE>();
        auto dataHasNan       = getDataHasNan<DTYPE>();
        //
        // compute block scale
        //
        double avg_scale = 0.0;
        int    n         = 0;
        for(size_t i = 0; i < (size_t)block_size; i++)
        {
            auto s = scales[i];
            auto d = data[i];

            // if NaN || Inf || Zero
            if(!isNaN<DTYPE>(reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0)
               && !isInf<DTYPE>(
                   reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0)
               && !isZero<DTYPE>(
                   reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0))
            {
                avg_scale += s;
                n++;
            }
        }

        avg_scale /= n;

        uint32_t block_scale = std::round(avg_scale);

        //
        // adjust data
        //
        for(size_t i = 0; i < (size_t)block_size; i++)
        {
            auto s = scales[i];
            auto d = data[i];

            // if NaN || Inf || Zero
            if(isNaN<DTYPE>(reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0))
            {
                if(!m_options.includeNaN /* Nan should not be generated */)
                    throw std::runtime_error("Internal Error");
                setNaN<DTYPE>(reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0);
            }
            else if(isInf<DTYPE>(
                        reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0))
            {
                if(!m_options.includeInf /* Inf should not be generated */)
                    throw std::runtime_error("Internal Error");
                setInf<DTYPE>(reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0);
            }
            else if(isZero<DTYPE>(
                        reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0))
            {
                setZero<DTYPE>(
                    reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0);
            }
            else
            {
                int32_t adjusted_s
                    = getExponentValue(d, dataMantissaBits, dataExponentBits) - dataBias;
                adjusted_s += s - block_scale;

                int32_t min_exp = dataUnbiasedEMin - dataMantissaBits;

                if(adjusted_s >= min_exp)
                {
                    // reference max normal
                    uint64_t max_normal;
                    setDataMax<DTYPE>(reinterpret_cast<uint8_t*>(&max_normal), 0);
                    double ref_max_normal
                        = toDouble<DTYPE>(reinterpret_cast<uint8_t*>(&block_scale),
                                          reinterpret_cast<uint8_t*>(&max_normal),
                                          0,
                                          0);
                    double ref_value = std::abs(toDouble<DTYPE>(
                        reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0));

                    // subnormal
                    if(adjusted_s < dataUnbiasedEMin)
                    {
                        // calc mantissa
                        uint64_t res_mantissa = (d & ((uint64_t(1) << dataMantissaBits) - 1)) >> 1;
                        res_mantissa |= uint64_t(1) << (dataMantissaBits - 1);
                        res_mantissa >>= dataUnbiasedEMin - adjusted_s - 1;

                        // set exponent and reset mantissa
                        d &= ~((uint64_t(1) << (dataExponentBits + dataMantissaBits)) - 1);

                        // set mantissa
                        d |= res_mantissa;
                    }
                    // normal
                    else if(ref_value <= ref_max_normal)
                    {
                        // set exponent
                        const uint32_t result_exp = adjusted_s + dataBias;

                        const uint64_t mask = (uint64_t(1) << dataExponentBits) - 1;

                        d &= ~(mask << dataMantissaBits);
                        d |= (result_exp & mask) << dataMantissaBits;
                    }
                    // overflow
                    else
                    {
                        const auto sign
                            = d & (uint64_t(1) << (dataExponentBits + dataMantissaBits));

                        if(m_options.includeInf && dataHasInf)
                        {
                            setInf<DTYPE>(reinterpret_cast<uint8_t*>(&s),
                                          reinterpret_cast<uint8_t*>(&d),
                                          0,
                                          0);
                        }
                        else if(m_options.includeNaN && dataHasNan)
                        {
                            setNaN<DTYPE>(reinterpret_cast<uint8_t*>(&s),
                                          reinterpret_cast<uint8_t*>(&d),
                                          0,
                                          0);
                        }
                        else
                        {
                            setDataMax<DTYPE>(reinterpret_cast<uint8_t*>(&d), 0);
                        }

                        d = sign | (d & ~(uint64_t(1) << (dataExponentBits + dataMantissaBits)));
                    }
                }
                else
                {
                    setZero<DTYPE>(
                        reinterpret_cast<uint8_t*>(&s), reinterpret_cast<uint8_t*>(&d), 0, 0);
                }
            }

            data[i] = d;
        }

        return block_scale;
    }

    template <typename DTYPE>
    uint32_t DataGenerator<DTYPE>::dispatch_scale_block(const std::vector<uint32_t>& scales,
                                                        std::vector<uint64_t>&       data,
                                                        int                          block_size)
    {
        switch(m_options.scaling)
        {
        case Mean:
            return scale_block_mean(scales, data, block_size);
        }

        return 0;
    }

    template <typename DTYPE>
    void DataGenerator<DTYPE>::post_sprinkle(const std::vector<int>& size,
                                             const std::vector<int>& stride,
                                             int32_t                 unbiased_min_exp)
    {
        const auto block_size = (isScaled<DTYPE>() ? m_options.blockScaling : size[0]);

        dimension_iterator ditr(size);

        bool do_nan = m_options.includeNaN;
        bool do_inf = m_options.includeInf;
        bool do_sbn = m_options.forceDenorm;

        if(!do_nan && !do_inf && !do_sbn)
        {
            return;
        }

        const size_t clmp_block_size
            = std::clamp(block_size, {SPRINKLE_BLOCK_MIN}, SPRINKLE_BLOCK_MAX);
        std::uniform_int_distribution<> idx_dist(0, clmp_block_size - 1);

        uint8_t temp_scale;

        //
        // For each block or collection of contiguous elements:
        //  - select a random element to set to NaN if applicable.
        //  - select a random element to set to Inf if applicable.
        //  - select a random element to set to a denorm if applicable and possible.
        //
        bool has_nan [[maybe_unused]] = false;
        bool has_inf [[maybe_unused]] = false;
        bool has_sbn                  = false;

        std::vector<bool> marked(clmp_block_size);
        size_t            marked_count = 0;
        size_t            block_data_i = 0;

        for(auto itr = ditr.begin(); itr != ditr.end(); ++itr)
        {
            size_t data_i       = (size_t)get_strided_idx(*itr, stride);
            size_t scale_i      = (data_i / block_size);
            size_t clmp_block_i = data_i % clmp_block_size;

            // reset
            if(clmp_block_i == 0)
            {
                has_nan = !do_nan;
                has_inf = !do_inf;
                has_sbn = !do_sbn;

                std::fill(marked.begin(), marked.end(), false);
                marked_count = 0;
                block_data_i = data_i;
            }

            // mark values of importance
            if(!has_nan)
            {
                has_nan = isNaN<DTYPE>(m_scaleBytes.data(), m_dataBytes.data(), scale_i, data_i);

                if(has_nan)
                {
                    marked[clmp_block_i] = true;
                    marked_count++;
                }
            }
            if(!has_inf)
            {
                has_inf = isInf<DTYPE>(m_scaleBytes.data(), m_dataBytes.data(), scale_i, data_i);

                if(has_inf)
                {
                    marked[clmp_block_i] = true;
                    marked_count++;
                }
            }
            if(!has_sbn)
            {
                has_sbn = isSubnorm<DTYPE>(m_dataBytes.data(), data_i);

                if(has_sbn)
                {
                    marked[clmp_block_i] = true;
                    marked_count++;
                }
            }

            // fill block with values
            if(clmp_block_i == clmp_block_size - 1)
            {
                if(!has_nan && clmp_block_size > marked_count)
                {
                    auto target = (size_t)idx_dist(m_gen);

                    while(marked[target])
                        target = (target + 1) % clmp_block_size;

                    setNaN<DTYPE>(&temp_scale, m_dataBytes.data(), 0, block_data_i + target);

                    marked[target] = true;
                    marked_count++;
                }
                if(!has_inf && clmp_block_size > marked_count)
                {
                    auto target = (size_t)idx_dist(m_gen);

                    while(marked[target])
                        target = (target + 1) % clmp_block_size;

                    setInf<DTYPE>(&temp_scale, m_dataBytes.data(), 0, block_data_i + target);

                    marked[target] = true;
                    marked_count++;
                }
                if(!has_sbn && clmp_block_size > marked_count)
                {
                    auto target = (size_t)idx_dist(m_gen);
                    while(marked[target])
                        target = (target + 1) % clmp_block_size;
                    const auto target_data_i  = block_data_i + target;
                    const auto target_scale_i = target_data_i / block_size;

                    // get scale
                    uint32_t biased_scale = 0;
                    if(isScaled<DTYPE>())
                        std::memcpy(&biased_scale,
                                    &m_scaleBytes[target_scale_i * m_scaleDesc.byte_size],
                                    m_scaleDesc.byte_size);

                    int32_t unbiased_scale = biased_scale - getScaleBias<DTYPE>();
                    int32_t exp_bound      = unbiased_scale + getDataUnBiasedEMin<DTYPE>();

                    if(unbiased_min_exp < exp_bound)
                    {
                        // get sign
                        uint64_t result = 0;
                        std::memcpy(&result,
                                    &m_dataBytes[target_data_i * m_dataDesc.byte_size],
                                    m_dataDesc.byte_size);
                        result &= (uint64_t(1) << (getDataMantissaBits<DTYPE>()
                                                   + getDataExponentBits<DTYPE>()));

                        // generate mantissa
                        uint64_t mantissa = 0;

                        uint64_t mantissa_dist_min = 1;
                        uint64_t mantissa_dist_max
                            = (uint64_t(1) << getDataMantissaBits<DTYPE>()) - 1;

                        if(unbiased_min_exp
                           > int32_t(exp_bound - int32_t(getDataMantissaBits<DTYPE>())))
                        {
                            mantissa_dist_min = 1 << int32_t(int32_t(getDataMantissaBits<DTYPE>())
                                                             + unbiased_min_exp - exp_bound);
                        }

                        std::uniform_int_distribution<> man_dist(mantissa_dist_min,
                                                                 mantissa_dist_max);

                        mantissa = man_dist(m_gen);

                        // assemble result
                        result |= mantissa;
                        std::memcpy(&m_dataBytes[target_data_i * m_dataDesc.byte_size],
                                    &result,
                                    m_dataDesc.byte_size);

                        marked[target] = true;
                        marked_count++;
                    }
                }
            }
        }
    }
}
