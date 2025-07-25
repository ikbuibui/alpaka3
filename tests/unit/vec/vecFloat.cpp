/* Copyright 2025 Tapish Narwal
 * SPDX-License-Identifier: MPL-2.0
 */
#include <alpaka/alpaka.hpp>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>
#include <catch2/matchers/catch_matchers_templated.hpp>

#include <cmath> // For std::abs, std::max
#include <concepts> // For std::floating_point
#include <cstdint>
#include <limits> // For std::numeric_limits
#include <sstream> // For std::ostringstream
#include <string>

/** @file
 *
 *  This file is testing vec floating point functionality
 */

// A Catch2 matcher for comparing alpaka::Vec with a hybrid tolerance.
// abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))
namespace Catch
{
    template<std::floating_point T, uint32_t Dim>
    struct VecApproxMatcher : Matchers::MatcherBase<alpaka::Vec<T, Dim>>
    {
        VecApproxMatcher(alpaka::Vec<T, Dim> const& expected, T rel_epsilon, T abs_epsilon)
            : m_expected(expected)
            , m_rel_epsilon(rel_epsilon)
            , m_abs_epsilon(abs_epsilon)
        {
        }

        bool match(alpaka::Vec<T, Dim> const& actual) const override
        {
            for(uint32_t i = 0; i < Dim; ++i)
            {
                if(std::abs(actual[i] - m_expected[i])
                   > std::max(m_abs_epsilon, m_rel_epsilon * std::max(std::abs(actual[i]), std::abs(m_expected[i]))))
                {
                    return false;
                }
            }
            return true;
        }

        std::string describe() const override
        {
            std::ostringstream ss;
            ss << "should be approximately equal to " << m_expected.toString() << " with relative epsilon "
               << m_rel_epsilon << " and absolute epsilon " << m_abs_epsilon;
            return ss.str();
        }

    private:
        alpaka::Vec<T, Dim> const& m_expected;
        T m_rel_epsilon;
        T m_abs_epsilon;
    };

    // A small relative epsilon is a sensible default.
    // A zero absolute epsilon means the check is purely relative for non-zero numbers.
    // You can provide a small value like 1e-8 if you need to compare against zero
    template<typename T, uint32_t Dim>
    [[nodiscard]] inline auto ApproxVec(
        alpaka::Vec<T, Dim> const& expected,
        T rel_epsilon = std::numeric_limits<T>::epsilon() * 100,
        T abs_epsilon = T{0.0})
    {
        return VecApproxMatcher<T, Dim>(expected, rel_epsilon, abs_epsilon);
    }

} // namespace Catch

/** Runtime test cases for floating point vectors */
struct FloatingPointRuntimeKernel
{
    // This kernel is not marked ALPAKA_FN_HOST_ACC as it uses Catch2 macros
    // which are only available in host-side runtime tests.
    void operator()() const
    {
        using namespace alpaka;

        SECTION("2D Floating Point Operations")
        {
            // Test with float, using a suitable epsilon for single precision.
            {
                using type = float;
                constexpr type epsilon = 1e-6f;
                auto const v1 = Vec<type, 2>{0.5f, -1.5f};
                auto const v2 = Vec<type, 2>{2.0f, 3.5f};
                type const s = 2.5f;

                CHECK_THAT(v1 + v2, Catch::ApproxVec(Vec<type, 2>{2.5f, 2.0f}, epsilon));
                CHECK_THAT(v1 - v2, Catch::ApproxVec(Vec<type, 2>{-1.5f, -5.0f}, epsilon));
                CHECK_THAT(v1 * v2, Catch::ApproxVec(Vec<type, 2>{1.0f, -5.25f}, epsilon));
                CHECK_THAT(v2 / v1, Catch::ApproxVec(Vec<type, 2>{4.0f, -2.333333f}, epsilon));

                CHECK_THAT(v1 + s, Catch::ApproxVec(Vec<type, 2>{3.0f, 1.0f}, epsilon));
                CHECK_THAT(s + v1, Catch::ApproxVec(Vec<type, 2>{3.0f, 1.0f}, epsilon));
                CHECK_THAT(v1 * s, Catch::ApproxVec(Vec<type, 2>{1.25f, -3.75f}, epsilon));
                CHECK_THAT(s * v1, Catch::ApproxVec(Vec<type, 2>{1.25f, -3.75f}, epsilon));
                CHECK_THAT(v1 / s, Catch::ApproxVec(Vec<type, 2>{0.2f, -0.6f}, epsilon));
            }

            // Test with double, using a tighter epsilon for double precision.
            {
                using type = double;
                constexpr type epsilon = 1e-12;
                auto const v1 = Vec<type, 2>{0.5, -1.5};
                auto const v2 = Vec<type, 2>{2.0, 3.5};

                CHECK_THAT(v1 + v2, Catch::ApproxVec(Vec<type, 2>{2.5, 2.0}, epsilon));
                CHECK_THAT(v1 - v2, Catch::ApproxVec(Vec<type, 2>{-1.5, -5.0}, epsilon));
                CHECK_THAT(v1 * v2, Catch::ApproxVec(Vec<type, 2>{1.0, -5.25}, epsilon));
                CHECK_THAT(v2 / v1, Catch::ApproxVec(Vec<type, 2>{4.0, -2.3333333333333335}, epsilon));
            }
        }

        SECTION("3D Floating Point Operations")
        {
            // Test with float
            {
                using type = float;
                constexpr type epsilon = 1e-6f;
                auto const v1 = Vec<type, 3>{0.5f, -1.5f, 10.2f};
                auto const v2 = Vec<type, 3>{2.0f, 3.5f, -0.1f};

                CHECK_THAT(v1 + v2, Catch::ApproxVec(Vec<type, 3>{2.5f, 2.0f, 10.1f}, epsilon));
                CHECK_THAT(v1 * v2, Catch::ApproxVec(Vec<type, 3>{1.0f, -5.25f, -1.02f}, epsilon));
                CHECK_THAT(v2 / v1, Catch::ApproxVec(Vec<type, 3>{4.0f, -2.333333f, -0.00980392f}, epsilon));
            }
        }

        SECTION("Precise Floating Point Operations")
        {
            // Test with float
            {
                using type = float;
                constexpr type epsilon = 1e-6f;
                auto const v1 = Vec<type, 2>{0.500000000000000000001111111e-18f, 10.2e-18f};
                auto const v2 = Vec<type, 2>{2.000000001e-18f, -0.1e-18f};

                CHECK_THAT(v1 + v2, Catch::ApproxVec(Vec<type, 2>{2.5e-18f, 10.1e-18f}, epsilon));
                CHECK_THAT(v1 * v2, Catch::ApproxVec(Vec<type, 2>{1.0e-36f, -1.02e-36f}, epsilon));
                CHECK_THAT(v2 / v1, Catch::ApproxVec(Vec<type, 2>{4.0f, -0.00980392}, epsilon));
            }
        }

        SECTION("Floating Point Comparisons")
        {
            // Comparison operators should yield exact boolean results, so no tolerance is needed here.
            using type = float;
            auto const v1 = Vec<type, 2>{1.0f, 2.0f};
            auto const v2 = Vec<type, 2>{1.5f, 1.5f};

            CHECK((v1 < v2) == Vec<bool, 2>{true, false});
            CHECK((v1 <= v2) == Vec<bool, 2>{true, false});
            CHECK((v1 > v2) == Vec<bool, 2>{false, true});
            CHECK((v1 >= v2) == Vec<bool, 2>{false, true});

            type const s = 1.5f;
            CHECK((v1 < s) == Vec<bool, 2>{true, false});
            CHECK((s > v1) == Vec<bool, 2>{true, false});
        }
    }
};

TEST_CASE("vec floats", "[vectorFloats]")
{
    using namespace alpaka;
    FloatingPointRuntimeKernel{}();
}
