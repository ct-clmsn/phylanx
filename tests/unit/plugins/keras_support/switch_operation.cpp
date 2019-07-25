// Copyright (c) 2019 Bita Hasheminezhad
// Copyright (c) 2019 Hartmut Kaiser
//
// Distributed under the Boost Software License, Version 1.0. (See accompanying
// file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#include <phylanx/phylanx.hpp>

#include <hpx/hpx_main.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/testing.hpp>

#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <blaze/Math.h>

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
void test_switch_operation(std::string const& code,
    std::string const& expected_str)
{
    HPX_TEST_EQ(compile_and_run(code), compile_and_run(expected_str));
}

///////////////////////////////////////////////////////////////////////////////
int main(int argc, char* argv[])
{
    test_switch_operation("switch(0, 42, 33)", "33.");
    test_switch_operation("switch(0, [42, 43], [13, 33])", "[13., 33.]");
    test_switch_operation("switch([10, 0, 0], [42., 43., 12.], [5., 13., 33.])",
        "[42., 13., 33.]");
    test_switch_operation("switch(true, [[42, 43],[1, 2]], [[13, 33],[3, 4]])",
        "[[42., 43.],[1., 2.]]");
    test_switch_operation("switch([1, -2, 0], [[10, 20],[30, 40],[50, 60]], "
                          "[[-10, -20],[-30, -40],[-50, -60]])",
        "[[ 10.,  20.],[ 30.,  40.],[-50., -60.]]");
    test_switch_operation(
        "switch([[1, 0],[1, 0],[0, 1]], [[10, 20],[30, 40],[50, 60]], "
        "[[-10, -20],[-30, -40],[-50, -60]])",
        "[[ 10., -20.],[ 30., -40.],[-50.,  60.]]");
    test_switch_operation("switch([[1, 0]], [[10, 20],[30, 40],[50, 60]], "
                          "[[-10, -20],[-30, -40],[-50, -60]])",
        "[[ 10., -20.],[ 30., -40.],[ 50., -60.]]");
    test_switch_operation("switch([[1],[1],[0]], [[10, 20],[30, 40],[50, 60]], "
                          "[[-10, -20],[-30, -40],[-50, -60]])",
        "[[ 10.,  20.],[ 30.,  40.],[-50., -60.]]");

#if defined(PHYLANX_HAVE_BLAZE_TENSOR)
    test_switch_operation("switch(0, [[[10,20]],[[30,40]],[[50,60]]], "
                          "[[[-10,-20]],[[-30,-40]],[[-50,-60]]])",
        "[[[-10., -20.]],[[-30., -40.]],[[-50., -60.]]]");
    test_switch_operation("switch([0, 3, 3], [[[10,20]],[[30,40]],[[50,60]]], "
                          "[[[-10,-20]],[[-30,-40]],[[-50,-60]]])",
        "[[[-10., -20.]],[[ 30.,  40.]],[[ 50.,  60.]]]");
    test_switch_operation(
        "switch([[0,3,0],[1,1,1]], "
        "[[[10,20,30,40],[50,60,70,80],[90,100,110,120]],"
        "[[1,2,3,4],[5,6,7,8],[9,10,11,12]]], "
        "[[[-10,-20,-30,-40],[-50,-60,-70,-80],[-90,-100,-110,-120]],"
        "[[-1,-2,-3,-4],[-5,-6,-7,-8],[-9,-10,-11,-12]]]"
        ")",
        "[[[ -10.,  -20.,  -30.,  -40.],[  50.,   60.,   70.,   80.], "
        "[ -90.,-100., -110., -120.]],[[   1.,    2.,    3.,    4.],  "
        "[   5.,    6.,  7.,    8.],[   9.,   10.,   11.,   12.]]]");
    test_switch_operation(
        "switch([[0.,3.]], "
        "[[[10,20,30,40],[50,60,70,80]],[[1,2,3,4],[5,6,7,8]]], "
        "[[[-10,-20,-30,-40],[-50,-60,-70,-80]],[[-1,-2,-3,-4],[-5,-6,-7,-8]]]"
        ")",
        "[[[-10., -20., -30., -40.],[ 50.,  60.,  70.,  80.]],  "
        "[[ -1.,  -2.,  -3.,  -4.],[  5.,   6.,   7.,   8.]]]");
    test_switch_operation(
        "switch([[0., 3.],[1., 1.]], "
        "[[[10,20,30,40],[50,60,70,80]],[[1,2,3,4],[5,6,7,8]]], "
        "[[[-10,-20,-30,-40],[-50,-60,-70,-80]],[[-1,-2,-3,-4],[-5,-6,-7,-8]]]"
        ")",
        "[[[-10., -20., -30., -40.],[ 50.,  60.,  70.,  80.]],"
        "[[  1.,   2.,   3.,   4.],[  5.,   6.,   7.,   8.]]]");
    test_switch_operation(
        "switch([[0],[3]],[[[10,20,30,40],[50,60,70,80],[90,100,110,120]],"
        "[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],[[[-10,-20,-30,-40],"
        "[-50,-60,-70,-80],[-90,-100,-110,-120]],[[-1,-2,-3,-4],"
        "[-5,-6,-7,-8],[-9,-10,-11,-12]]])",
        "[[[-10.,-20.,-30.,-40.],[-50.,-60.,-70.,-80.],[-90.,-100.,-110.,-120.]"
        "],"
        "[[1., 2.,  3.,  4.], [ 5., 6., 7., 8.],[ 9., 10., 11., 12.]]]");
    test_switch_operation(
        "switch([[[-2,0,1,-2]]],[[[10,20,30,40],[50,60,70,80],[90,100,110,120]]"
        ",[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],[[[-10,-20,-30,-40],"
        "[-50,-60,-70,-80],[-90,-100,-110,-120]],[[-1,-2,-3,-4],"
        "[-5,-6,-7,-8],[-9,-10,-11,-12]]])",
        "[[[  10.,  -20.,   30.,   40.],[  50.,  -60.,   70.,   80.],"
        "[  90., -100.,  110.,  120.]],[[   1.,   -2.,    3.,    4.],"
        "[   5.,   -6.,    7.,    8.],[   9.,  -10.,   11.,   12.]]]");
    test_switch_operation(
        "switch([[[2],[0],[1]]],[[[10,20,30,40],[50,60,70,80],[90,100,110,120]]"
        ",[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],[[[-10,-20,-30,-40],"
        "[-50,-60,-70,-80],[-90,-100,-110,-120]],[[-1,-2,-3,-4],"
        "[-5,-6,-7,-8],[-9,-10,-11,-12]]])",
        "[[[ 10.,  20.,  30.,  40.],[-50., -60., -70., -80.],"
        "[ 90., 100., 110., 120.]],[[  1.,   2.,   3.,   4.],"
        "[ -5.,  -6.,  -7.,  -8.],[  9.,  10.,  11.,  12.]]]");
    test_switch_operation(
        "switch([[[2]],[[0]]], [[[10,20,30,40],[50,60,70,80],[90,100,110,120]]"
        ",[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],[[[-10,-20,-30,-40],"
        "[-50,-60,-70,-80],[-90,-100,-110,-120]],[[-1,-2,-3,-4],"
        "[-5,-6,-7,-8],[-9,-10,-11,-12]]])",
        "[[[ 10.,  20.,  30.,  40.],[ 50.,  60.,  70.,  80.],"
        "[ 90., 100., 110., 120.]],[[ -1.,  -2.,  -3.,  -4.],"
        "[ -5.,  -6.,  -7.,  -8.],[ -9., -10., -11., -12.]]]");
    test_switch_operation(
        "switch([[[2.,0.,0.,1.],[1.,1.,0.,0.],[0.,42.,0.,0.]]],"
        "[[[10,20,30,40],[50,60,70,80],[90,100,110,120]]"
        ",[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],[[[-10,-20,-30,-40],"
        "[-50,-60,-70,-80],[-90,-100,-110,-120]],[[-1,-2,-3,-4],"
        "[-5,-6,-7,-8],[-9,-10,-11,-12]]])",
        "[[[  10.,  -20.,  -30.,   40.],[  50.,   60.,  -70.,  -80.],"
        "[ -90.,  100., -110., -120.]],[[   1.,   -2.,   -3.,    4.],"
        "[   5.,    6.,   -7.,   -8.],[  -9.,   10.,  -11.,  -12.]]]");
    test_switch_operation(
        "switch([[[2.,0.,0.,1.]],[[1.,1.,0.,0.]]],"
        "[[[10,20,30,40],[50,60,70,80],[90,100,110,120]]"
        ",[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],[[[-10,-20,-30,-40],"
        "[-50,-60,-70,-80],[-90,-100,-110,-120]],[[-1,-2,-3,-4],"
        "[-5,-6,-7,-8],[-9,-10,-11,-12]]])",
        "[[[  10.,  -20.,  -30.,   40.],[  50.,  -60.,  -70.,   80.],"
        "[  90., -100., -110.,  120.]],[[   1.,    2.,   -3.,   -4.],"
        "[   5.,    6.,   -7.,   -8.],[   9.,   10.,  -11.,  -12.]]]");
    test_switch_operation(
        "switch([[[2.],[0.],[0.]],[[1.],[1.],[0.]]],"
        "[[[10,20,30,40],[50,60,70,80],[90,100,110,120]]"
        ",[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],[[[-10,-20,-30,-40],"
        "[-50,-60,-70,-80],[-90,-100,-110,-120]],[[-1,-2,-3,-4],"
        "[-5,-6,-7,-8],[-9,-10,-11,-12]]])",
        "[[[  10.,   20.,   30.,   40.],[ -50.,  -60.,  -70.,  -80.],"
        "[ -90., -100., -110., -120.]],[[   1.,    2.,    3.,    4.],"
        "[   5.,    6.,    7.,    8.],[  -9.,  -10.,  -11.,  -12.]]]");
    test_switch_operation(
        "switch([[[1,0,1,0],[0,0,0,1],[0,1,0,1]],"
        "[[0,0,0,0],[1,0,1,1],[1,1,1,1]]],"
        "[[[10,20,30,40],[50,60,70,80],[90,100,110,120]]"
        ",[[1,2,3,4],[5,6,7,8],[9,10,11,12]]],[[[-10,-20,-30,-40],"
        "[-50,-60,-70,-80],[-90,-100,-110,-120]],[[-1,-2,-3,-4],"
        "[-5,-6,-7,-8],[-9,-10,-11,-12]]])",
        "[[[  10.,  -20.,   30.,  -40.],[ -50.,  -60.,  -70.,   80.],"
        "[ -90.,  100., -110.,  120.]],[[  -1.,   -2.,   -3.,   -4.],"
        "[   5.,   -6.,    7.,    8.],[   9.,   10.,   11.,   12.]]]");

#endif

    return hpx::util::report_errors();
}
