//  Copyright (c) 2018 Christopher Taylor
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
//
#if !defined(__PHYLANX_RANDOMFORESTIMPL_HPP__)
#define __PHYLANX_RANDOMFORESTIMPL_HPP__

namespace phylanx { namespace algorithms { namespace impl {

struct tsne {

    static void neg_squared_euc_distance(blaze::DynamicMatrix<double> & X
        , blaze::DynamicMatrix<double> & D) {

        using citerator = blaze::DynamicMatrix<double>::ConstIterator;

        blaze::DynamicVector<double> sum_X(X.rows(), 0.0);

        {
            auto row_indices =
                boost::irange<std::uint64_t>(0UL, X.size());

            hpx::parallel::algorithms::for_each(
                hpx::parallel::execution::par
                , std::begin(row_indices)
                , std::end(row_indices)
                , [&X, &sum_X](auto const i) {
                    sum_X[i] = std::reduce(
                        X.cbegin(i)
                        , X.cend(i)
                        , 0.0
                        , [](auto const& x, auto const& y) {
                            return x->value() + y->value();
                        });
                });
        }

        D = trans( (-2.0 * (X * trans(X))) + sum_X) + sum_X;
        D = -D;
    }

    static softmax(X, diag_zero=true) {
        //auto ex = exp(X - 
    }

    static void fit() {
    }

    static void predict() {
    }

};

}}}
