/* Copyright 2024 René Widera
 * SPDX-License-Identifier: MPL-2.0
 */

#include <alpaka/alpaka.hpp>
#include <alpaka/example/executeForEach.hpp>
#include <alpaka/example/executors.hpp>

#include <catch2/catch_template_test_macros.hpp>
#include <catch2/catch_test_macros.hpp>

#include <chrono>
#include <functional>
#include <iostream>
#include <thread>

/** @file In this file we test if the a compile time vector is forwarded correctly into the kernel without silent
 * conversions into a runtime value.
 */

using namespace alpaka;
using namespace alpaka::onHost;

using TestApis = std::decay_t<decltype(allExecutorsAndApis(enabledApis))>;

struct KernelCVec
{
    template<typename T>
    ALPAKA_FN_ACC void operator()(T const& acc, auto result) const
    {
        static_assert(result()[0] == 42u);
    }
};

TEMPLATE_LIST_TEST_CASE("CVec frame extent kernel call", "", TestApis)
{
    auto cfg = TestType::makeDict();
    auto api = cfg[object::api];
    auto exec = cfg[object::exec];

    std::cout << api.getName() << std::endl;

    Platform platform = makePlatform(api);
    Device device = platform.makeDevice(0);

    std::cout << getName(platform) << " " << device.getName() << std::endl;

    Queue queue = device.makeQueue();


    newCVec::CVec<Vec{42}> cudaWorks;
    Platform cpuPlatform = makePlatform(api::cpu);
    Device cpuDevice = cpuPlatform.makeDevice(0);
    wait(queue);
    {
        onHost::enqueue(queue, exec, FrameSpec{Vec{1u}, Vec<uint32_t, 1>{43u}}, KernelBundle{KernelCVec{}, cudaWorks});
    }
}
