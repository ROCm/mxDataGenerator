/*******************************************************************************
 *
 * MIT License
 *
 * Copyright 2024-2025 AMD ROCm(TM) Software
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 *******************************************************************************/

#pragma once

#include "dataTypeInfo.hpp"
#include "data_generation_utils.hpp"

#include <cmath>
#include <limits>
#include <random>
#include <string>
#include <thread>
#include <variant>

#include <omp.h>

namespace DGen
{
    constexpr uint64_t ONE = 1;

    constexpr index_t SPRINKLE_BLOCK_MIN = 3;
    constexpr index_t SPRINKLE_BLOCK_MAX = 15;

    //
    // Defining Data Initialization Modes
    //
    struct Bounded
    {
        std::string toString() const
        {
            return "Bounded";
        }
        friend std::ostream& operator<<(std::ostream& os, const Bounded& b)
        {
            return os << b.toString();
        }
    };

    struct BoundedAlternatingSign
    {
        std::string toString() const
        {
            return "BoundedAlternatingSign";
        }
        friend std::ostream& operator<<(std::ostream& os, const BoundedAlternatingSign& bas)
        {
            return os << bas.toString();
        }
    };

    struct Unbounded
    {
        std::string toString() const
        {
            return "Unbounded";
        }
        friend std::ostream& operator<<(std::ostream& os, const Unbounded& u)
        {
            return os << u.toString();
        }
    };

    struct IdentityScale_NormalData
    {
        float mean;
        float std_dev;

        std::string toString() const
        {
            return "IdentityScale_NormalData(" + std::to_string(mean) + ", "
                   + std::to_string(std_dev) + ")";
        }
        friend std::ostream& operator<<(std::ostream& os, const IdentityScale_NormalData& isnd)
        {
            return os << isnd.toString();
        }
    };

    struct NormalScale_UniformData
    {
        float mean;
        float std_dev;

        std::string toString() const
        {
            return "NormalScale_UniformData(" + std::to_string(mean) + ", "
                   + std::to_string(std_dev) + ")";
        }
        friend std::ostream& operator<<(std::ostream& os, const NormalScale_UniformData& nsud)
        {
            return os << nsud.toString();
        }
    };

    struct Identity
    {
        std::string toString() const
        {
            return "Identity";
        }
        friend std::ostream& operator<<(std::ostream& os, const Identity& i)
        {
            return os << i.toString();
        }
    };

    struct Ones
    {
        std::string toString() const
        {
            return "Ones";
        }
        friend std::ostream& operator<<(std::ostream& os, const Ones& o)
        {
            return os << o.toString();
        }
    };

    struct Zeros
    {
        std::string toString() const
        {
            return "Zeros";
        }
        friend std::ostream& operator<<(std::ostream& os, const Zeros& z)
        {
            return os << z.toString();
        }
    };

    using RawDataInitMode = std::variant<Bounded,
                                         BoundedAlternatingSign,
                                         Unbounded,
                                         IdentityScale_NormalData,
                                         NormalScale_UniformData,
                                         Identity,
                                         Ones,
                                         Zeros>;

    struct Trigonometric
    {
        std::string toString() const
        {
            return "Trigonometric";
        }
        friend std::ostream& operator<<(std::ostream& os, const Trigonometric& t)
        {
            return os << t.toString();
        }
    };

    struct Normal
    {
        float mean;
        float std_dev;

        std::string toString() const
        {
            return "Normal(" + std::to_string(mean) + ", " + std::to_string(std_dev) + ")";
        }
        friend std::ostream& operator<<(std::ostream& os, const Normal& n)
        {
            return os << n.toString();
        }
    };

    using FloatDataInitMode = std::variant<Trigonometric, Normal>;

    using DataInitMode = std::variant<RawDataInitMode, FloatDataInitMode>;

    template <class... Ts>
    struct overload : Ts...
    {
        using Ts::operator()...;
    };

    template <class... Ts>
    overload(Ts...) -> overload<Ts...>;

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

        DataInitMode init_mode = RawDataInitMode(Bounded{});

        double min = -1.0;
        double max = 1.0;

        DataScaling scaling      = DataScaling::Mean;
        index_t     blockScaling = 1;
    };

    template <typename DTYPE>
    class DataGenerator
    {
    public:
        DataGenerator() = default;

        using Generator = std::mt19937;
        // generate internal byte buffers/
        DataGenerator& generate(std::vector<index_t>        size,
                                std::vector<index_t>        stride,
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

        uint                   m_seed = 1713573848;
        std::vector<Generator> m_gen;
        const int              m_num_threads = std::min(32, omp_get_max_threads());

        struct BufferDesc
        {
            index_t array_size  = 0;
            index_t bit_size    = 0;
            index_t byte_size   = 0;
            index_t buffer_size = 0;
        };

        BufferDesc m_dataDesc;
        BufferDesc m_scaleDesc;

        std::vector<uint8_t> m_dataBytes;
        std::vector<uint8_t> m_scaleBytes;

        static std::vector<uint8_t> packArray(BufferDesc in_desc, const std::vector<uint8_t>& src);

        void dispatch_generate_data(const std::vector<index_t>& size,
                                    const std::vector<index_t>& stride);

        void generate_data_bounded(const std::vector<index_t>& size);
        void generate_data_bounded_alternating_sign(const std::vector<index_t>& size);
        void generate_data_unbounded(const std::vector<index_t>& size);
        void generate_data_trigonometric(const std::vector<index_t>& size);
        void generate_data_normal(const std::vector<index_t>& size,
                                  const float                 mean,
                                  const float                 std_dev);
        void generate_data_identity(const std::vector<index_t>& size,
                                    const std::vector<index_t>& stride);
        void generate_data_ones();

        void dispatch_generate_pattern(const std::vector<index_t>& size,
                                       const std::vector<index_t>& stride);

        uint32_t scale_block_mean(const std::vector<uint32_t>& scales,
                                  std::vector<uint64_t>&       data,
                                  index_t                      block_size);
        uint32_t dispatch_scale_block(const std::vector<uint32_t>& scales,
                                      std::vector<uint64_t>&       data,
                                      index_t                      block_size);

        void post_sprinkle(const std::vector<index_t>& size, int32_t unbiased_min_exp);

        void setGenerator(int numThreads);
    };
}

#include <DataGenerator_impl.hpp>
