//  Copyright (c) 2017 Alireza Kheirkhahan
//
//  Distributed under the Boost Software License, Version 1.0. (See accompanying
//  file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PHYLANX_PRIMITIVES_FILE_READ_CSV_OCT_26_2017_0129PM)
#define PHYLANX_PRIMITIVES_FILE_READ_CSV_OCT_26_2017_0129PM

#include <phylanx/config.hpp>
#include <phylanx/execution_tree/primitives/base_primitive.hpp>

#include <hpx/include/components.hpp>

#include <string>
#include <vector>

namespace phylanx { namespace execution_tree { namespace primitives
{
    class HPX_COMPONENT_EXPORT file_read_csv
      : public base_primitive
      , public hpx::components::component_base<file_read_csv>
    {
    public:
        static std::vector<match_pattern_type> const match_data;

        file_read_csv() = default;

        file_read_csv(std::vector<primitive_argument_type>&& operands);

        hpx::future<primitive_result_type> eval(
            std::vector<primitive_argument_type> const& args) const override;

    private:
        std::string filename_;
    };
}}}

#endif
