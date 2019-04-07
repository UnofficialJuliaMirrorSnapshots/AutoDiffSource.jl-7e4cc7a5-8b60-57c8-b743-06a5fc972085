function checkdiff(f, δf, x0...)
    x = [x0...]
    y0 = f(x...)
    @assert length(y0) == 1 "Scalar functions only"
    y, ∇f = δf(x...)
    @assert isapprox(y0, y) "Return values do not match"
    #    @assert typeof(y0) === typeof(y) "Return type doesn't match"
    ∂x = ∇f()
    #    @assert typeof(x0) === typeof(∂x) "Gradient type doesn't match"
    checkgrad(f, x, ∂x)
end

function checkgrad(f, x, ∂x, h = 1e-8, atol = 1e-8)
    for k = 1:length(x)
        ∂xₖ = length(x) == 1 ? ∂x : ∂x[k]
        if isa(x[k], AbstractFloat)
            x1 = deepcopy(x)
            x1[k] -= h
            y1 = f(x1...)
            x2 = deepcopy(x)
            x2[k] += h
            y2 = f(x2...)
            isapprox(2h * ∂xₖ, y2-y1, atol=atol) || error("gradient for argument #$k doesn't match by $((y2-y1)/(2h*∂xₖ))")
        elseif isa(x[k], AbstractArray)
            for l = eachindex(x[k])
                x1 = deepcopy(x)
                x1[k][l] -= h
                y1 = f(x1...)
                x2 = deepcopy(x)
                x2[k][l] += h
                y2 = f(x2...)
                isapprox(2h * ∂xₖ[l], y2-y1, atol=atol) || error("gradient for argument #$k element $l doesn't match by $((y2-y1)/(2h*∂xₖ[l]))")
            end
        else error("not supported argument #$k type: $(typeof(x[k]))")
        end
    end
    true
end
