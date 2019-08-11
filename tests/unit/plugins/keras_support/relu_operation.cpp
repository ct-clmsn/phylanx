// Copyright (c) 2019 Shahrzad Shirzad
// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/phylanx.hpp>

#include <hpx/hpx_main.hpp>
#include <hpx/testing.hpp>

#include <cstdint>
#include <string>
#include <utility>

///////////////////////////////////////////////////////////////////////////////
phylanx::execution_tree::primitive_argument_type compile_and_run(
    std::string const& codestr)
{
    phylanx::execution_tree::compiler::function_list snippets;
    phylanx::execution_tree::compiler::environment env =
        phylanx::execution_tree::compiler::default_environment();

    auto const& code = phylanx::execution_tree::compile(codestr, snippets, env);
    return code.run();
}

///////////////////////////////////////////////////////////////////////////////
void test_relu(std::string const& code, std::string const& expected_str)
{
    HPX_TEST_EQ(compile_and_run(code), compile_and_run(expected_str));
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    test_relu("relu(0.2, 0.5, 2., 0.5)", "-0.15");
    test_relu("relu(0.2)", "0.2");
    test_relu("relu(0.7, 0.5, 2., 0.5)", "0.7");
    test_relu("relu(2.2, 0.5, 2., 0.5)", "2.0");
    test_relu("relu(-0.7)", "0.0");
    test_relu("relu(2.2, 0.5, nil, 0.5)", "2.2");
    test_relu("relu(2.2, 2., 1., 1.2)", "1.0");
    test_relu("relu(-2.2, 2.)", "-4.4");
    test_relu("relu(2.2, 2., 1.)", "1.0");
    test_relu("relu(-1.8, 2., 1., 1.2)", "-6.0");
    test_relu("relu(-1.8, 2., nil, 1.2)", "-6.0");

    test_relu("relu([1., -2., 6.], 0.5, 2., 0.5)", "[1., -1.25, 2.]");
    test_relu("relu([1., -2., 6.], 2., 0.5, 3.)", "[-4., -10., 0.5]");
    test_relu("relu([1., -2., 6.], 0.5, nil, 0.5)", "[1., -1.25, 6.]");
    test_relu("relu([3., 0.5, 1.5], 0.5, 2, 0.5)", "[2., 0.5, 1.5]");
    test_relu("relu([3., 0.5, 1.5], 2.5, 1.5, 4.)", "[-2.5, -8.75, -6.25]");
    test_relu("relu([3., 0.5, 1.5], 2.5, nil, 4.)", "[-2.5, -8.75, -6.25]");

    test_relu("relu([[1., -2., 6.], [3.0, -0.2, 2.]])",
        "[[1., 0., 6.], [3., 0., 2.]]");
    test_relu("relu([[1., -2., 6.], [3.0, -0.2, 2.]], 0.5)",
        "[[1., -1., 6.], [3., -0.1, 2.]]");
    test_relu("relu([[1., -2., 6.], [3.0, -0.2, 2.]], 0.5, 5.)",
        "[[1., -1., 5.], [3., -0.1, 2.]]");
    test_relu("relu([[1., -2., 6.], [3.0, -0.2, 2.]], 0.5, 5., 2.)",
        "[[-0.5, -2., 5.], [3., -1.1, 2.]]");
    test_relu("relu([[1., -2., 6.], [3.0, -0.2, 2.]], 4., 2., 3.)",
        "[[-8., -20., 2.], [2., -12.8, -4.]]");
    test_relu("relu([[1., -2., 6.], [3.0, -0.2, 2.]], 4., nil, 3.)",
        "[[-8., -20., 6.], [3., -12.8, -4.]]");
    test_relu("relu([[-1.5, 3., 2.5], [1.0, 0.2, 8.]], 4., 2., 3.)",
        "[[-18., 2., -2.], [-8., -11.2, 2.]]");
    test_relu("relu([[-1.5, 3., 2.5], [1.0, 0.2, 8.]], 0.5, 5., 2.)",
        "[[-1.75, 3., 2.5], [-0.5, -0.9, 5.]]");
    test_relu("relu([[-1.5, 3., 2.5], [1.0, 0.2, 8.]], 0.5, nil, 2.)",
        "[[-1.75, 3., 2.5], [-0.5, -0.9, 8.]]");

    test_relu("relu([[[1., -2., 6.], [3.0, -0.2, 2.]], [[-1., 3., 5.], [1.0, "
              "-0.4, 8.]]])",
        "[[[1., 0., 6.], [3., 0., 2.]], [[0., 3., 5.], [1., 0.0,  8.]]]");
    test_relu("relu([[[1., -2., 6.], [3.0, -0.2, 2.]], [[-1., 3., 5.], [1.0, "
              "-0.4, 8.]]], 0.5)",
        "[[[1., -1., 6.], [3, -0.1, 2]], [[-0.5, 3., 5.], [1., -0.2,  8.]]]");
    test_relu("relu([[[1., -2., 6.], [3.0, -0.2, 2.]], [[-1., 3., 5.], [1.0, "
              "-0.4, 8.]]], 0.5, 5.)",
        "[[[1., -1., 5.], [3., -0.1, 2.]], [[-0.5, 3., 5.], [1., -0.2,  5.]]]");
    test_relu("relu([[[1., -2., 6.], [3.0, -0.2, 2.]], [[-1., 3., 5.], [1.0, "
              "-0.4, 8.]]], 0.5, 5., 2.)",
        "[[[-0.5, -2, 5], [3, -1.1, 2]], [[-1.5, 3, 5], [-0.5, -1.2,  5.]]]");
    test_relu("relu([[[1., -2., 6.], [3.0, -0.2, 2.]], [[-1., 3., 5.], [1.0, "
              "-0.4, 8.]]], 4., 2., 3.)",
        "[[[-8., -20., 2.], [2., -12.8, -4.]], [[-16., 2., 2.], [-8., -13.6,  "
        "2.]]]");
    test_relu("relu([[[1., -2., 6.], [3.0, -0.2, 2.]], [[-1., 3., 5.], [1.0, "
              "-0.4, 8.]]], 4., nil, 3.)",
        "[[[-8., -20., 6.], [3., -12.8, -4.]], [[-16., 3., 5.], [-8., -13.6,  "
        "8.]]]");
    test_relu("relu([[[-1.5, 3., 2.5], [1.0, 0.2, 8.]], [[1.0, -0.4, 8.], "
              "[-0.5, -2., 5.]]], 0.5, 5, 2)",
        "[[[-1.75, 3., 2.5], [-0.5, -0.9, 5.]], [[-0.5, -1.2, 5.], [-1.25, "
        "-2., 5.]]]");
    test_relu("relu([[[-1.5, 3., 2.5], [1.0, 0.2, 8.]], [[1.0, -0.4, 8.], "
              "[-0.5, -2., 5.]]], 4., 2., 3.)",
        "[[[-18., 2., -2.], [-8., -11.2, 2.]], [[-8., -13.6, 2.], [-14., -20., "
        " 2.]]]");
    test_relu("relu([[[-1.5, 3., 2.5], [1.0, 0.2, 8.]], [[1.0, -0.4, 8.], "
              "[-0.5, -2., 5.]]], 4., nil, 3.)",
        "[[[-18., 3., -2.], [-8., -11.2, 8.]], [[-8., -13.6, 8.], [-14., -20., "
        " 5.]]]");

    return hpx::util::report_errors();
}
