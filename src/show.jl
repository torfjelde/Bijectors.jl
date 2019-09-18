import Base: show

# the name of a distribution
#
#   Generally, this should be just the type name, e.g. Normal.
#   Under certain circumstances, one may want to specialize
#   this function to provide a name that is easier to read,
#   especially when the type is parametric.
#
function bijectorname(d::B) where {N, B<:Bijector{N}}
    tname = string(nameof(B))
    pstring = (B.parameters[end] == N) ? join(B.parameters[1:end - 1], ", ") : join(B.parameters[1:end], ", ")
    if length(pstring) > 40
        pstring = "..."
    end
    return tname * "{" * pstring * ", Dims=$N}"
end
# bijectorname(d::Composed{A, Dim}) where {A, Dim} = join((nameof(typeof(d)), ""))

show(io::IO, d::Bijector) = show(io, d, fieldnames(typeof(d)))

# For some distributions, the fields may contain internal details,
# which we don't want to show, this function allows one to
# specify which fields to show.
#
function show(io::IO, d::Bijector, pnames)
    uml, namevals = _use_multline_show(d, pnames)
    uml ? show_multline(io, d, namevals) : show_oneline(io, d, namevals)
end

const _NameVal = Tuple{Symbol,Any}

function _use_multline_show(d::Bijector, pnames)
    # decide whether to use one-line or multi-line format
    #
    # Criteria: if total number of values is greater than 8, or
    # there are matrix-valued params, we use multi-line format
    #
    namevals = _NameVal[]
    multline = false
    tlen = 0
    for (i, p) in enumerate(pnames)
        pv = getfield(d, p)
        if !(isa(pv, Number) || isa(pv, NTuple) || isa(pv, AbstractVector))
            multline = true
        else
            tlen += length(pv)
        end
        push!(namevals, (p, pv))
    end
    if tlen > 8
        multline = true
    end
    return (multline, namevals)
end

function _use_multline_show(d::Bijector)
    _use_multline_show(d, fieldnames(typeof(d)))
end

function show_oneline(io::IO, d::Bijector, namevals)
    print(io, bijectorname(d))
    np = length(namevals)
    print(io, '(')
    for (i, nv) in enumerate(namevals)
        (p, pv) = nv
        print(io, p)
        print(io, '=')
        show(io, pv)
        if i < np
            print(io, ", ")
        end
    end
    print(io, ')')
end

function show_multline(io::IO, d::Bijector, namevals)
    print(io, bijectorname(d))
    println(io, "(")
    for (p, pv) in namevals
        print(io, p)
        print(io, ": ")
        println(io, pv)
    end
    println(io, ")")
end


# using Bijectors: Exp, Logit, PlanarLayer
# Bijectors.Exp()
# Bijectors.Logit(0.0, 1.0)
# Bijectors.composel([PlanarLayer(10) for i = 1:3]...)
