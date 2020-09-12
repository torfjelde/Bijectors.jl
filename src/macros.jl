using MacroTools

function _get_varname(expr)
    if MacroTools.@capture(expr, x_::T_)
        x
    else
        expr
    end
end

"""
    @deftransform function transform(b::Bijector, x) ... end

Essentially results in

    function transform(b::Bijector, x) ... end
    (b::Bijector)(x) = transform(b, x)
"""
macro deftransform(expr)
    def = MacroTools.splitdef(expr)

    # First arg should be the `Bijector`.
    # Second arg should be the input.
    bijector_def, input_def = def[:args]
    body = def[:body]
    name = esc(def[:name])

    # extract the input variables
    input_var = _get_varname(input_def)
    bijector_var = _get_varname(bijector_def)

    return quote
        $(Base).@__doc__ function $(name)($bijector_def, $input_def)
            $body
        end

        ($bijector_def)($input_def) = $(name)($bijector_var, $input_var)
    end
end


### Sharing computation ###
# function forward_clean(expr)
#     if Meta.isexpr(expr, :macrocall) && length(expr.args) == 3
#         if (expr.args[1] === Symbol("@transform")) || (expr.args[1] === Symbol("@logabsdetjac"))
#             return expr.args[3]
#         else
#             return expr
#         end
#     else
#         return expr
#     end
# end

"""
    @bijector function f(b::Bijector, x) ... end

Takes the method `forward` and uses it to define both `transform`
and `logabsdetjac`, while ensuring that any shared computation is
taken advantage of in `forward`.
"""
macro bijector(expr)
    def = MacroTools.splitdef(expr)
    body = def[:body]
    args = def[:args]
    whereparams = def[:whereparams]

    # extract the input variables
    bijector_arg, input_arg = args

    # Define the `(b::Bijector)(x::T)` signature
    bijector_call_expr = if !isempty(whereparams)
        quote
            ($bijector_arg)($input_arg) where {$whereparams...} = $(Bijectors).transform($bijector_arg, $input_arg)
        end
    else
        quote
            ($bijector_arg)($input_arg) = $(Bijectors).transform($bijector_arg, $input_arg)
        end
    end

    # Figure out what is shared, and what isn't
    shared_exprs = [] # beginning of `forward`
    transform_exprs = []
    logjac_exprs = []
    tail_exprs = [] # goes at the end of `forward`

    for expr in body.args
        # When using `rv = ...` and `logabsdetjac = ...`
        if Meta.isexpr(expr, :(=)) && expr.args[1] == :rv
            push!(transform_exprs, Expr(:return, expr.args[2]))
            push!(tail_exprs, expr)
        elseif Meta.isexpr(expr, :(=)) && expr.args[1] == :logabsdetjac
            push!(logjac_exprs, Expr(:return, expr.args[2]))
            push!(tail_exprs, expr)
        else
            # If still sharing, add those expressions
            issharing = (length(transform_exprs) == 0) && (length(logjac_exprs) == 0)
            if issharing
                push!(shared_exprs, expr)
            else
                push!(tail_exprs, expr)
            end
        end
    end

    # Add the shared computation
    transform_full_exprs = copy(shared_exprs)
    append!(transform_full_exprs, transform_exprs)

    logjac_full_exprs = copy(shared_exprs)
    append!(logjac_full_exprs, logjac_exprs)

    # Remove the redundant macro's from the `forward` body
    forward_full_exprs = vcat(shared_exprs, tail_exprs)

    push!(forward_full_exprs, Expr(:return, :(rv = rv, logabsdetjac = logabsdetjac)))

    # HACK: `esc` because the types of the arguments are getting the namespace of `Bijectors`
    # because this is the namespace it's expanded in. Not sure how if `esc` everything is
    # the best solution.
    return esc(quote
        function $(Bijectors).transform($(args...))
            $(transform_full_exprs...)
        end
        $bijector_call_expr

        function $(Bijectors).logabsdetjac($(args...))
            $(logjac_full_exprs...)
        end

        function $(Bijectors).forward($(args...))
            $(forward_full_exprs...)
        end
    end)
end

