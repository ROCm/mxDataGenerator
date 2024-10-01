#pragma once

#include <iostream>
#include <tuple>
#include <vector>


namespace DGen
{

    template <typename CartesianProductVectorType, typename... Ts>
    auto cartesian_product_helper(CartesianProductVectorType& vec, std::tuple<Ts...> t)
    {
        vec.emplace_back(t);
    }

    template <typename CartesianProductVectorType,
              typename... Ts,
              typename Iter,
              typename... TailIters>
    auto cartesian_product_helper(CartesianProductVectorType& vec,
                                  std::tuple<Ts...>           t,
                                  Iter                        b,
                                  Iter                        e,
                                  TailIters... tail_iters)
    {
        for(auto iter = b; iter != e; ++iter)
            cartesian_product_helper(vec, std::tuple_cat(t, std::make_tuple(*iter)), tail_iters...);
    }

    template <typename CartesianProductVectorType, typename... Iters>
    auto cartesian_product(CartesianProductVectorType& vec, Iters... iters)
    {
        cartesian_product_helper(vec, std::make_tuple(), iters...);
    }

    struct dimension_iterator
    {
        const std::vector<int> dimensions;
        dimension_iterator(std::vector<int> dim)
            : dimensions(dim)
        {
        }

        struct iterator
        {
        private:
            const std::vector<int> dimensions;
            std::vector<int>       current;
            std::vector<int>*      curr_ptr;

        public:
            iterator(std::vector<int> dim)
                : dimensions(dim)
                , current(dim.size(), 0)
                , curr_ptr(&current)
            {
            }

            iterator()
                : dimensions(0)
                , current(0)
                , curr_ptr(nullptr)
            {
            }

            void print()
            {
                for(const auto e : current)
                {
                    std::cout << e << " ";
                }
                std::cout << "\n";
            }

            const std::vector<int>& operator*() const
            {
                return *curr_ptr;
            }
            const std::vector<int>* operator->()
            {
                return curr_ptr;
            }

            // Prefix increment
            iterator& operator++()
            {
                for(int i = 0; i < dimensions.size(); ++i)
                {
                    current[i] += 1;
                    if(current[i] >= dimensions[i])
                    {
                        current[i] = 0;
                    }
                    else
                    {
                        return *this;
                    }
                }

                curr_ptr = nullptr;
                return *this;
            }

            // Postfix increment
            iterator operator++(int)
            {
                iterator tmp = *this;
                ++(*this);
                return tmp;
            }

            friend bool operator==(const iterator& a, const iterator& b)
            {
                if(a.curr_ptr == nullptr || b.curr_ptr == nullptr)
                {
                    return a.curr_ptr == b.curr_ptr;
                }
                return *a.curr_ptr == *b.curr_ptr;
            }
            friend bool operator!=(const iterator& a, const iterator& b)
            {
                return !(a == b);
            }
        };

        iterator begin() const
        {
            return iterator(dimensions);
        }
        iterator end() const
        {
            return iterator();
        }
    };

    inline int get_strided_idx(const std::vector<int>& indices, const std::vector<int>& stride)
    {
        int res = 0;
        for(int i = 0; i < indices.size(); i++)
        {
            res += indices[i] * stride[i];
        }

        return res;
    }

    inline void split_dynamic(uint64_t  input,
                              uint32_t  exp_size,
                              uint32_t  man_size,
                              uint8_t*  sign,
                              uint32_t* exponent,
                              uint64_t* mantissa)
    {
        if(sign)
            *sign = input >> (exp_size + man_size);
        if(exponent)
            *exponent = (input >> man_size) & ((1UL << exp_size) - 1UL);
        if(mantissa)
            *mantissa = input & ((1ULL << man_size) - 1ULL);
    }

    inline void split_double(double val, uint8_t* sign, uint32_t* exponent, uint64_t* mantissa)
    {
        union
        {
            double   input;
            uint64_t output;
        } bit_rep;

        bit_rep.input = val;

	split_dynamic(bit_rep.output, 11u, 52u, sign, exponent, mantissa);
    }
}
