__precompile__()
module AutoDiffSource

export @δ, δ, checkdiff, checkgrad

export δplus, δminus, δtimes, δdivide, δabs, δsum, δmean, δsqrt, δexp, δlog, δpower, δdot
export δdot_plus, δdot_minus, δdot_times, δdot_divide, δdot_abs, δdot_sqrt, δdot_exp, δdot_log, δdot_power
export δplus_const1, δminus_const1, δtimes_const1, δdivide_const1, δpower_const1
export δdot_plus_const1, δdot_minus_const1, δdot_times_const1, δdot_divide_const1, δdot_power_const1
export δplus_const2, δminus_const2, δtimes_const2, δdivide_const2, δpower_const2
export δdot_plus_const2, δdot_minus_const2, δdot_times_const2, δdot_divide_const2, δdot_power_const2
export δfanout, δzeros, δzeros_const, δones_const, δlength_const, δcolon_const, δref_const2, δtuple
export δsrand_const, δrand_const, δrandn_const, δsize_const, δsign_const

export δlog1p, δexpm1, δsin, δcos, δtan, δsinh, δcosh, δtanh, δasin, δacos, δatan
export δround_const, δfloor_const, δceil_const, δtrunc_const, δmod2pi, δmaximum, δminimum, δtranspose
export δerf, δerfc, δgamma, δlgamma, δmin, δmax, δmin_const1, δmax_const1, δmin_const2, δmax_const2

export δdot_min, δdot_max, δdot_min_const1, δdot_max_const1, δdot_min_const2, δdot_max_const2
export δmulticast, δmulticast_const1, δmulticast_const2

if VERSION >= v"0.6-"
    using SpecialFunctions
end

include("parse.jl")
include("diff.jl")
include("func.jl")
include("checkdiff.jl")
include("multicast.jl")

macro δ(expr::Expr)
    esc(:( $expr; $(δ(macroexpand(expr)))))
end

macro δ(f::Symbol)
    fs = methods(eval(GlobalRef(Main, f)))
    length(fs) >  0 || error("function '$f' not found")
    expr = Expr(:block)
    for fdef in fs.ms
        fn = VERSION >= v"0.6.0-" ? Base.uncompressed_ast(fdef, fdef.source).code : Base.uncompressed_ast(fdef.lambda_template)
        fcode = fn[2:end]
        info = Expr(:line, fdef.line, fdef.file)
        types = getfield(fdef.sig, 3)
        fargs = [Expr(:(::), Symbol("_$i"), types[i]) for i in 2:length(types)]
        fname = Symbol(string(f))
        func = :(function $fname($(fargs...)); end)
        body = func.args[2].args
        empty!(body)
        foreach(arg -> push!(body, arg), fcode)
        push!(expr.args, δ(func, info))
    end
    esc(expr)
end

function δ(expr, info = Expr(:line))
    ops = parse_function(expr, info)
    for op in ops.body
        name = replace(string(op.name), r"^\.", "")
        if !isdefined(Symbol("δ$name")) && !isdefined(Symbol("δ$(name)_const"))
            eval(Main, :(@δ $(Symbol(name))))
        end
    end

    ex = Expr(:block)
    push!(ex.args, delta(ops))
    ins = filter(isvar, ops.inputs)
    if length(ins) > 1
        for name in ins
            if isa(name, Expr)
                name = name.args[1]
            end
            op = parse_function(expr, info, Dict{Symbol,Symbol}(name => Symbol(string(name) * "_const")))
            push!(ex.args, delta(op))
        end
        if length(ins) > 2
            for name1 in ins
                if isa(name1, Expr)
                    name1 = name1.args[1]
                end
                for name2 in ins
                    if isa(name2, Expr)
                        name2 = name2.args[1]
                    end
                    if name2 > name1
                        op = parse_function(expr, info, Dict{Symbol,Symbol}(name1 => Symbol(string(name1) * "_const"),
                                                                            name2 => Symbol(string(name2) * "_const")))
                        push!(ex.args, delta(op))
                    end
                end
            end
        end
    end
    ex
end

end
