isvar(n::Number) = false
isvar(n::Symbol) = !endswith(string(n), "_const")
isvar(n::Expr) = n.head == :(::) ? isvar(n.args[1]) : true
isconst(n) = !isvar(n)

const reversenames = Dict(:times => :(*), :plus => :(+), :divide => :(/), :minus => :(-), :power => :(^),
                          :dot_times => :(.*), :dot_plus => :(.+), :dot_divide => :(./), :dot_minus => :(.-), :dot_power => :(.^))

function delta(ops)
    funcname = Symbol("δ$(ops.name)")
    func = :(function $funcname($(ops.inputs...)); end)
    body = func.args[2].args::Vector{Any}
    empty!(body)
    nablas = Dict()
    last_info = [Expr(:line)]
    for line in ops.body
        push_if_changed!(body, last_info, line.info)
        name = replace(string(line.name), r"^\.", "")
        constname = Symbol("δ$(name)_const")
        if line.name == :ref_const2
            nablas[line] = (:ref_const2, line.inputs[2])
            push!(body, :($(toexpr(line.outputs)) = $(line.inputs[1])[$(line.inputs[2])]))
        elseif isdefined(constname) || all(isconst, line.inputs) || all(isconst, line.outputs)
            sname = Symbol(name)
            sname = get(reversenames, sname, sname)
            if sname == :fanout
                temp = gensym(name)
                push!(body, :($temp = $(line.inputs...)))
                [push!(body, :($(line.outputs[k]) = $temp[$k])) for k in 1:length(line.outputs)]
            elseif startswith(string(line.name), ".")
                push!(body, :($(toexpr(line.outputs)) = $sname.($(line.inputs...))))
            else
                push!(body, :($(toexpr(line.outputs)) = $sname($(line.inputs...))))
            end
        else
            nabla = gensym("∇" * name)
            nablas[line] = nabla
            temp = gensym(name)
            funcname = Symbol("δ" * name)
            # push!(body, :(($(line.outputs...), $nabla) = $name($(line.inputs...)))),
            # work around for https://github.com/JuliaLang/julia/issues/15276
            if startswith(string(line.name), ".")
                mcast = Symbol("δ$(name_const("multicast", line.inputs, line.outputs))")
                push!(body, :($temp = $mcast($funcname, $(line.inputs...))))
            else
                push!(body, :($temp = $funcname($(line.inputs...))))
            end
            [push!(body, :($(line.outputs[k]) = $temp[$k])) for k in 1:length(line.outputs)]
            push!(body, :($nabla = $temp[$(length(line.outputs)+1)]))
            # end work around
        end
    end
    push!(body, ∇(ops, nablas))
    push!(body, :($(ops.outputs...), $(Symbol("∇$(ops.name)"))))
    #    @show func
    func
end

function ∇(ops, nablas)
    name = Symbol("∇$(ops.name)")
    inputs = map(topartial, filter(isvar, ops.outputs))
    func = :(function $name($(inputs...)); end)
    if length(inputs) == 1
        func.args[1].args[2] = Expr(:kw, func.args[1].args[2], 1.0)
    end
    body = func.args[2].args::Vector{Any}
    empty!(body)
    dupes = Set(inputs)
    emptys = Set()
    last_info = [Expr(:line)]
    for line in reverse(ops.body)
        push_if_changed!(body, last_info, line.info)
        if haskey(nablas, line)
            nabla = nablas[line]
            if isa(nabla, Tuple) && nabla[1] == :ref_const2
                ins = map(topartial, filter(isvar, line.outputs))
                outs = map(topartial, filter(isvar, line.inputs))
                if outs[1] in dupes
                    push!(body, :($(outs[1])[$(nabla[2])] += $(toexpr(ins))))
                else
                    push!(body, :($(outs[1]) = δzeros($(line.inputs[1]))))
                    push!(body, :($(outs[1])[$(nabla[2])] = $(toexpr(ins))))
                    push!(dupes, outs[1])
                end
            else
                ins = map(topartial, filter(isvar, line.outputs))
                outs = map(topartial, filter(isvar, line.inputs))
                if length(outs) > 0
                    for o in filter(isvar, line.outputs)
                        op = topartial(o)
                        if op in emptys && !(op in dupes)
                            push!(body, :($op = δzeros($o)))
                        end
                    end
                    dedup = [k in dupes ? gensym(k) : (push!(dupes, k); k) for k in outs]
                    push!(body, :($(toexpr(dedup)) = $nabla($(ins...))))
                    [push!(body, :($(outs[k]) += $(dedup[k]))) for k in find(outs .!= dedup)]
                end
            end
        else
            [push!(emptys, topartial(o)) for o in filter(isvar, line.inputs)]
        end
    end
    push!(body, toexpr(map(topartial, filter(isvar, ops.inputs))))
    func
end

topartial(expr::Symbol) = Symbol("∂$expr")
topartial(expr::Expr) = Symbol("∂$(expr.args[1])")
toexpr(symbols) = length(symbols) == 1 ? symbols[1] : Expr(:tuple, symbols...)

function push_if_changed!(body, last_info, info)
    if last_info[1] != info
        last_info[1] = info
        push!(body, info)
    end
end
