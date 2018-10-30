// Copyright (c) 2017 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)


#include <phylanx/phylanx.hpp>

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/util/lightweight_test.hpp>

#include <array>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

void test_constant_0d()
{
    phylanx::execution_tree::primitive val =
        phylanx::execution_tree::primitives::create_variable(
            hpx::find_here(), phylanx::ir::node_data<double>(42.0));

    phylanx::execution_tree::primitive const_ =
        phylanx::execution_tree::primitives::create_constant(
            hpx::find_here(),
            phylanx::execution_tree::primitive_arguments_type{
                std::move(val)
            });

    hpx::future<phylanx::execution_tree::primitive_argument_type> f =
        const_.eval();

    auto result = phylanx::execution_tree::extract_numeric_value(f.get());

    HPX_TEST_EQ(result.num_dimensions(), 0);
    HPX_TEST_EQ(42.0, result[0]);
}

void test_constant_1d()
{
    phylanx::execution_tree::primitive val =
        phylanx::execution_tree::primitives::create_variable(
            hpx::find_here(), phylanx::ir::node_data<double>(42.0));

    phylanx::execution_tree::primitive const_ =
        phylanx::execution_tree::primitives::create_constant(
            hpx::find_here(),
            phylanx::execution_tree::primitive_arguments_type{
                std::move(val), phylanx::ir::node_data<std::int64_t>(1007)
            });

    hpx::future<phylanx::execution_tree::primitive_argument_type> f =
        const_.eval();

    blaze::DynamicVector<double> expected =
        blaze::DynamicVector<double>(1007UL, 42.0);
    auto result = phylanx::execution_tree::extract_numeric_value(f.get());

    HPX_TEST_EQ(result.num_dimensions(), 1);
    HPX_TEST_EQ(result.dimension(0), 1007);
    HPX_TEST_EQ(phylanx::ir::node_data<double>(std::move(expected)), result);
}

void test_constant_2d()
{
    phylanx::execution_tree::primitive val =
        phylanx::execution_tree::primitives::create_variable(
            hpx::find_here(), phylanx::ir::node_data<double>(42.0));

    phylanx::execution_tree::primitive const_ =
        phylanx::execution_tree::primitives::create_constant(hpx::find_here(),
            phylanx::execution_tree::primitive_arguments_type{
                std::move(val),
                phylanx::execution_tree::primitive_argument_type{
                    phylanx::execution_tree::primitive_arguments_type{
                        phylanx::ir::node_data<std::int64_t>(105),
                        phylanx::ir::node_data<std::int64_t>(101)}
                    }});

    hpx::future<phylanx::execution_tree::primitive_argument_type> f =
        const_.eval();

    blaze::DynamicMatrix<double> expected =
        blaze::DynamicMatrix<double>(105UL, 101UL, 42.0);
    auto result = phylanx::execution_tree::extract_numeric_value(f.get());

    HPX_TEST_EQ(result.num_dimensions(), 2);
    HPX_TEST_EQ(result.dimension(0), 105);
    HPX_TEST_EQ(result.dimension(1), 101);
    HPX_TEST_EQ(phylanx::ir::node_data<double>(std::move(expected)), result);
}

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

void test_constant_operation(std::string const& code,
    std::string const& expected_str)
{
    HPX_TEST_EQ(compile_and_run(code), compile_and_run(expected_str));
}

void test_empty_operation(std::string const& code,
    std::array<int, 2> const& dims)
{
    auto f = compile_and_run(code);
    auto result_dims =
        phylanx::execution_tree::extract_numeric_value_dimensions(f());

    HPX_TEST_EQ(dims[0], result_dims[0]);
    HPX_TEST_EQ(dims[1], result_dims[1]);
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    test_constant_0d();
    test_constant_1d();
    test_constant_2d();

    // zeros, ones, full
    test_constant_operation("constant(42, list())", "42");
    test_constant_operation("constant(42, list(4))", "hstack(42, 42, 42, 42)");
    test_constant_operation(
        "constant(42, list(2, 2))", "hstack(vstack(42, 42), vstack(42, 42))");

    // ...like operations
    test_constant_operation("constant_like(42, 1)", "42");
    test_constant_operation("constant_like(42, hstack(1, 2, 3, 4))",
        "hstack(42, 42, 42, 42)");
    test_constant_operation(
        "constant_like(42, hstack(vstack(1, 2), vstack(3, 4)))",
        "hstack(vstack(42, 42), vstack(42, 42))");

    // empty
    test_empty_operation("constant__int(list())", {1, 1});
    test_empty_operation("constant__int(list(4))", {1, 4});
    test_empty_operation("constant__int(list(2, 2))", {2, 2});

    // empty_like
    test_empty_operation("constant_like__int(1)", {1, 1});
    test_empty_operation("constant_like__int(hstack(1, 2, 3, 4))", {1, 4});
    test_empty_operation(
        "constant_like__int(hstack(vstack(1, 2), vstack(3, 4)))", {2, 2});

    return hpx::util::report_errors();
}

