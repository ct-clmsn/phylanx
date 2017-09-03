//   Copyright (c) 2001-2011 Joel de Guzman
//   Copyright (c) 2001-2017 Hartmut Kaiser
//
//   Distributed under the Boost Software License, Version 1.0. (See accompanying
//   file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)

#if !defined(PHYLANX_AST_NODE_HPP)
#define PHYLANX_AST_NODE_HPP

#include <phylanx/config.hpp>
#include <phylanx/ast/parser/extended_variant.hpp>
#include <phylanx/ir/node_data.hpp>
#include <phylanx/util/optional.hpp>
#include <phylanx/util/serialization/optional.hpp>
#include <phylanx/util/serialization/variant.hpp>

#include <hpx/runtime/serialization/serialization_fwd.hpp>

#include <boost/spirit/include/support_extended_variant.hpp>
#include <boost/spirit/include/support_attributes.hpp>

#include <cstddef>
#include <iosfwd>
#include <list>
#include <string>
#include <utility>

namespace phylanx { namespace ast
{
    ///////////////////////////////////////////////////////////////////////////
    //  The AST
    struct tagged
    {
        std::size_t id; // Used to annotate the AST with the iterator position.
                        // This id is used as a key to a map<int, Iterator>
                        // (not really part of the AST.)
    };

    enum class optoken
    {
        op_unknown,

        // precedence 1
        op_comma,

        // precedence 2
        op_assign,
        op_plus_assign,
        op_minus_assign,
        op_times_assign,
        op_divide_assign,
        op_mod_assign,
        op_bit_and_assign,
        op_bit_xor_assign,
        op_bitor_assign,
        op_shift_left_assign,
        op_shift_right_assign,

        // precedence 3
        op_logical_or,

        // precedence 4
        op_logical_and,

        // precedence 5
        op_bit_or,

        // precedence 6
        op_bit_xor,

        // precedence 7
        op_bit_and,

        // precedence 8
        op_equal,
        op_not_equal,

        // precedence 9
        op_less,
        op_less_equal,
        op_greater,
        op_greater_equal,

        // precedence 10
        op_shift_left,
        op_shift_right,

        // precedence 11
        op_plus,
        op_minus,

        // precedence 12
        op_times,
        op_divide,
        op_mod,

        // precedence 13
        op_positive,
        op_negative,
        op_pre_incr,
        op_pre_decr,
        op_compl,
        op_not,

        // precedence 14
        op_post_incr,
        op_post_decr,
    };

    ///////////////////////////////////////////////////////////////////////////
    struct nil {};

    constexpr inline bool operator==(nil const&, nil const&)
    {
        return true;
    }
    constexpr inline bool operator!=(nil const&, nil const&)
    {
        return false;
    }

    ///////////////////////////////////////////////////////////////////////////
    struct identifier : tagged
    {
        identifier() = default;

        identifier(std::string const& name)
        : name(name)
        {
        }
        identifier(std::string && name)
          : name(std::move(name))
        {
        }

        std::string name;

        PHYLANX_EXPORT void serialize(
            hpx::serialization::input_archive& ar, unsigned);
        PHYLANX_EXPORT void serialize(
            hpx::serialization::output_archive& ar, unsigned);
    };

    inline bool operator==(identifier const& lhs, identifier const& rhs)
    {
        return lhs.name == rhs.name;
    }
    inline bool operator!=(identifier const& lhs, identifier const& rhs)
    {
        return !(lhs == rhs);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct unary_expr;
    struct expression;

//     struct function_call;
//     struct if_statement;
//     struct while_statement;
//     struct statement;
//     struct return_statement;
//
//     using statement_list = std::list<statement>;

    ///////////////////////////////////////////////////////////////////////////
    using expr_node_type = phylanx::ast::parser::extended_variant<
            nil
          , bool
          , phylanx::ir::node_data<double>
          , identifier
          , phylanx::util::recursive_wrapper<expression>
        >;

    struct primary_expr : tagged, expr_node_type
    {
        primary_expr() = default;

        primary_expr(nil val)
          : expr_node_type(val)
        {
        }

        primary_expr(bool val)
          : expr_node_type(val)
        {
        }

        primary_expr(phylanx::ir::node_data<double> const& val)
          : expr_node_type(val)
        {
        }
        primary_expr(phylanx::ir::node_data<double> && val)
          : expr_node_type(std::move(val))
        {
        }
        primary_expr(double val)
          : expr_node_type(phylanx::ir::node_data<double>(val))
        {
        }

        primary_expr(identifier const& val)
          : expr_node_type(val)
        {
        }
        primary_expr(identifier && val)
          : expr_node_type(std::move(val))
        {
        }
        primary_expr(std::string const& val)
          : expr_node_type(identifier(val))
        {
        }
        primary_expr(std::string && val)
          : expr_node_type(identifier(std::move(val)))
        {
        }

        primary_expr(expression const& val)
          : expr_node_type(val)
        {
        }
        primary_expr(expression && val)
          : expr_node_type(std::move(val))
        {
        }

        // this is only for usage in the Python bindings
        expr_node_type const& value() const { return *this; }

        PHYLANX_EXPORT void serialize(
            hpx::serialization::input_archive& ar, unsigned);
        PHYLANX_EXPORT void serialize(
            hpx::serialization::output_archive& ar, unsigned);
    };

    ///////////////////////////////////////////////////////////////////////////
    using operand_node_type = phylanx::ast::parser::extended_variant<
            nil
          , phylanx::util::recursive_wrapper<primary_expr>
          , phylanx::util::recursive_wrapper<unary_expr>
//           , phylanx::util::recursive_wrapper<function_call>
        >;

    struct operand : tagged, operand_node_type
    {
        operand() = default;

        operand(double val)
          : operand_node_type(
                phylanx::util::recursive_wrapper<primary_expr>(val))
        {
        }
        operand(std::string const& val)
          : operand_node_type(
                phylanx::util::recursive_wrapper<primary_expr>(val))
        {
        }
        operand(std::string && val)
          : operand_node_type(
                phylanx::util::recursive_wrapper<primary_expr>(std::move(val)))
        {
        }

        operand(primary_expr const& val)
          : operand_node_type(val)
        {
        }
        operand(primary_expr && val)
          : operand_node_type(std::move(val))
        {
        }

        operand(unary_expr const& val)
          : operand_node_type(val)
        {
        }
        operand(unary_expr && val)
          : operand_node_type(std::move(val))
        {
        }

//         operand(function_call const& val)
//            : operand_node_type(val)
//         {
//         }
//         operand(function_call && val)
//            : operand_node_type(std::move(val))
//         {
//         }

        // this is only for usage in the Python bindings
        operand_node_type const& value() const { return *this; }

        PHYLANX_EXPORT void serialize(
            hpx::serialization::input_archive& ar, unsigned);
        PHYLANX_EXPORT void serialize(
            hpx::serialization::output_archive& ar, unsigned);
    };

//     inline bool operator==(operand const& lhs, operand const& rhs)
//     {
//         return lhs.get() == rhs.get();
//     }
//     inline bool operator!=(operand const& lhs, operand const& rhs)
//     {
//         return !(lhs == rhs);
//     }

    ///////////////////////////////////////////////////////////////////////////
    struct unary_expr : tagged
    {
        unary_expr()
          : operator_(optoken::op_unknown)
        {}

        unary_expr(optoken id, operand const& op)
          : operator_(id)
          , operand_(op)
        {}
        unary_expr(optoken id, operand && op)
          : operator_(id)
          , operand_(std::move(op))
        {}

        optoken operator_;
        operand operand_;

        PHYLANX_EXPORT void serialize(
            hpx::serialization::input_archive& ar, unsigned);
        PHYLANX_EXPORT void serialize(
            hpx::serialization::output_archive& ar, unsigned);
    };

    inline bool operator==(unary_expr const& lhs, unary_expr const& rhs)
    {
        return lhs.operator_ == rhs.operator_ &&
            lhs.operand_ == rhs.operand_;
    }
    inline bool operator!=(unary_expr const& lhs, unary_expr const& rhs)
    {
        return !(lhs == rhs);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct operation
    {
        operation()
          : operator_(optoken::op_unknown)
        {}

        operation(optoken id, operand const& op)
          : operator_(id)
          , operand_(op)
        {}
        operation(optoken id, operand && op)
          : operator_(id)
          , operand_(std::move(op))
        {}

        optoken operator_;
        operand operand_;

        PHYLANX_EXPORT void serialize(
            hpx::serialization::input_archive& ar, unsigned);
        PHYLANX_EXPORT void serialize(
            hpx::serialization::output_archive& ar, unsigned);
    };

    inline bool operator==(operation const& lhs, operation const& rhs)
    {
        return lhs.operator_ == rhs.operator_ &&
            lhs.operand_ == rhs.operand_;
    }
    inline bool operator!=(operation const& lhs, operation const& rhs)
    {
        return !(lhs == rhs);
    }

    ///////////////////////////////////////////////////////////////////////////
    struct expression
    {
        expression() = default;

        expression(operand const& f)
          : first(f)
        {}
        expression(operand && f)
          : first(std::move(f))
        {}

        void append(operation const& op)
        {
            rest.push_back(op);
        }
        void append(operation && op)
        {
            rest.emplace_back(std::move(op));
        }
        void append(std::list<operation> const& l)
        {
            std::copy(l.begin(), l.end(), std::back_inserter(rest));
        }
        void append(std::list<operation> && l)
        {
            std::move(l.begin(), l.end(), std::back_inserter(rest));
        }

        operand first;
        std::list<operation> rest;

        PHYLANX_EXPORT void serialize(
            hpx::serialization::input_archive& ar, unsigned);
        PHYLANX_EXPORT void serialize(
            hpx::serialization::output_archive& ar, unsigned);
    };

    inline bool operator==(expression const& lhs, expression const& rhs)
    {
        return lhs.first == rhs.first && lhs.rest == rhs.rest;
    }
    inline bool operator!=(expression const& lhs, expression const& rhs)
    {
        return !(lhs == rhs);
    }

//     ///////////////////////////////////////////////////////////////////////////
//     struct function_call
//     {
//         identifier function_name;
//         std::list<expression> args;
//
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::input_archive& ar, unsigned);
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::output_archive& ar, unsigned);
//     };
//
//     inline bool operator==(function_call const& lhs, function_call const& rhs)
//     {
//         return lhs.function_name == rhs.function_name && lhs.args == rhs.args;
//     }
//     inline bool operator!=(function_call const& lhs, function_call const& rhs)
//     {
//         return !(lhs == rhs);
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     struct assignment
//     {
//         identifier lhs;
//         optoken operator_;
//         expression rhs;
//
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::input_archive& ar, unsigned);
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::output_archive& ar, unsigned);
//     };
//
//     inline bool operator==(assignment const& lhs, assignment const& rhs)
//     {
//         return lhs.lhs == rhs.lhs && lhs.operator_ == rhs.operator_ &&
//             lhs.rhs == rhs.rhs;
//     }
//     inline bool operator!=(assignment const& lhs, assignment const& rhs)
//     {
//         return !(lhs == rhs);
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     struct variable_declaration
//     {
//         identifier lhs;
//         phylanx::util::optional<expression> rhs;
//
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::input_archive& ar, unsigned);
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::output_archive& ar, unsigned);
//     };
//
//     inline bool operator==(
//         variable_declaration const& lhs, variable_declaration const& rhs)
//     {
//         return lhs.lhs == rhs.lhs && lhs.rhs == rhs.rhs;
//     }
//     inline bool operator!=(
//         variable_declaration const& lhs, variable_declaration const& rhs)
//     {
//         return !(lhs == rhs);
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     using statement_node_type = phylanx::ast::parser::extended_variant<
//             nil
//           , variable_declaration
//           , assignment
//           , phylanx::util::recursive_wrapper<if_statement>
//           , phylanx::util::recursive_wrapper<while_statement>
//           , phylanx::util::recursive_wrapper<return_statement>
//           , phylanx::util::recursive_wrapper<statement_list>
//           , phylanx::util::recursive_wrapper<expression>
//         >;
//
//     struct statement : statement_node_type
//     {
//         statement() = default;
//
//         statement(nil val)
//           : statement_node_type(val)
//         {
//         }
//
//         statement(variable_declaration const& val)
//           : statement_node_type(val)
//         {
//         }
//         statement(variable_declaration && val)
//           : statement_node_type(std::move(val))
//         {
//         }
//
//         statement(assignment const& val)
//           : statement_node_type(val)
//         {
//         }
//         statement(assignment && val)
//           : statement_node_type(std::move(val))
//         {
//         }
//
//         statement(if_statement const& val)
//           : statement_node_type(val)
//         {
//         }
//         statement(if_statement && val)
//           : statement_node_type(std::move(val))
//         {
//         }
//
//         statement(while_statement const& val)
//           : statement_node_type(val)
//         {
//         }
//         statement(while_statement && val)
//           : statement_node_type(std::move(val))
//         {
//         }
//
//         statement(return_statement const& val)
//           : statement_node_type(val)
//         {
//         }
//         statement(return_statement && val)
//           : statement_node_type(std::move(val))
//         {
//         }
//
//         statement(statement_list const& val)
//           : statement_node_type(val)
//         {
//         }
//         statement(statement_list && val)
//           : statement_node_type(std::move(val))
//         {
//         }
//
//         statement(expression const& val)
//           : statement_node_type(val)
//         {
//         }
//         statement(expression && val)
//           : statement_node_type(std::move(val))
//         {
//         }
//
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::input_archive& ar, unsigned);
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::output_archive& ar, unsigned);
//     };
//
//     ///////////////////////////////////////////////////////////////////////////
//     struct if_statement
//     {
//         expression condition;
//         statement then;
//         phylanx::util::optional<statement> else_;
//
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::input_archive& ar, unsigned);
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::output_archive& ar, unsigned);
//     };
//
//     inline bool operator==(if_statement const& lhs, if_statement const& rhs)
//     {
//         return lhs.condition == rhs.condition && lhs.then == rhs.then &&
//             lhs.else_ == rhs.else_;
//     }
//     inline bool operator!=(if_statement const& lhs, if_statement const& rhs)
//     {
//         return !(lhs == rhs);
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     struct while_statement
//     {
//         expression condition;
//         statement body;
//
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::input_archive& ar, unsigned);
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::output_archive& ar, unsigned);
//     };
//
//     inline bool operator==(
//         while_statement const& lhs, while_statement const& rhs)
//     {
//         return lhs.condition == rhs.condition && lhs.body == rhs.body;
//     }
//     inline bool operator!=(
//         while_statement const& lhs, while_statement const& rhs)
//     {
//         return !(lhs == rhs);
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     struct return_statement : tagged
//     {
//         phylanx::util::optional<expression> expr;
//
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::input_archive& ar, unsigned);
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::output_archive& ar, unsigned);
//     };
//
//     inline bool operator==(
//         return_statement const& lhs, return_statement const& rhs)
//     {
//         return lhs.expr == rhs.expr;
//     }
//     inline bool operator!=(
//         return_statement const& lhs, return_statement const& rhs)
//     {
//         return !(lhs == rhs);
//     }
//
//     ///////////////////////////////////////////////////////////////////////////
//     struct function
//     {
//         std::string return_type;
//         identifier function_name;
//         std::list<identifier> args;
//         phylanx::util::optional<statement_list> body;
//
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::input_archive& ar, unsigned);
//         PHYLANX_EXPORT void serialize(
//             hpx::serialization::output_archive& ar, unsigned);
//     };
//
//     using function_list = std::list<function>;

    ///////////////////////////////////////////////////////////////////////////
    // print functions for debugging
    PHYLANX_EXPORT std::ostream& operator<<(std::ostream& out, nil);
    PHYLANX_EXPORT std::ostream& operator<<(std::ostream& out, optoken op);
    PHYLANX_EXPORT std::ostream& operator<<(
        std::ostream& out, identifier const& id);
    PHYLANX_EXPORT std::ostream& operator<<(
        std::ostream& out, primary_expr const& p);
    PHYLANX_EXPORT std::ostream& operator<<(
        std::ostream& out, operand const& op);
    PHYLANX_EXPORT std::ostream& operator<<(
        std::ostream& out, unary_expr const& ue);
    PHYLANX_EXPORT std::ostream& operator<<(
        std::ostream& out, operation const& op);
    PHYLANX_EXPORT std::ostream& operator<<(
        std::ostream& out, expression const& expr);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, function_call const& fc);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, assignment const& assign);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, variable_declaration const& vd);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, statement const& stmt);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, if_statement const& if_);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, while_statement const& while_);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, return_statement const& ret);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, function const& func);
//     PHYLANX_EXPORT std::ostream& operator<<(
//         std::ostream& out, function_list const& fl);
}}

#endif
