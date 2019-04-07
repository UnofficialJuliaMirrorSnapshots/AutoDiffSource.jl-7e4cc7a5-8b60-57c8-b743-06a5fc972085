    using AutoDiffSource
using Base.Test

function checkdiff_inferred(f, δf, x0...)
    x = [x0...]
    y0 = f(x...)
    @assert length(y0) == 1 "Scalar functions only"
    y, ∇f = Test.@inferred δf(x...)
    @assert isapprox(y0, y) "Return values do not match"
    @assert typeof(y0) === typeof(y) "Return type doesn't match"
    ∂x = Test.@inferred ∇f(1.0)
    if isa(∂x, Tuple)
        @assert typeof(x0) === typeof(∂x) "Gradient type doesn't match: $(typeof(x0)) vs $(typeof(∂x))"
    else
        @assert typeof(x0) === typeof((∂x,)) "Gradient type doesn't match: : $(typeof(x0)) vs $(typeof((∂x,)))"
    end
    checkgrad(f, x, ∂x)
end


# Test example
sigmoid(x) = 1 / (1 + exp(-x))
function autoencoder(We1, We2, Wd, b1, b2, input)
    firstLayer = sigmoid.(We1 * input + b1)
    encodedInput = sigmoid.(We2 * firstLayer + b2)
    reconstructedInput = sigmoid.(Wd * encodedInput)
    return reconstructedInput
end
@δ function autoencoderError(We1, We2, Wd, b1, b2, input)
    reconstructedInput = autoencoder(We1, We2, Wd, b1, b2, input)
    return sum((input - reconstructedInput).^2)
end
@assert checkdiff_inferred(autoencoderError, δautoencoderError, randn(3,3), randn(3,3), rand(3,3), randn(3), randn(3), randn(3))

# Array indexing
@δ rosenbrock(x, y) = sum(100*(y-x.^2).^2 + (1-x).^2)
@δ function rosenbrock(x)
    l = length(x)
    rosenbrock(x[1:l-1], x[2:l])
end
@assert checkdiff_inferred(rosenbrock, δrosenbrock, randn(3))

# diffentiate a 3rd party function without source code, recursively
@δ rosen2(x::Vector{Float64}, y::Vector{Float64}) = sum(100*(y-x.^2).^2 + (1-x).^2)
function rosen(x::Vector{Float64})
    l = length(x)
    rosen2(x[1:l-1], x[2:l])
end
@δ rosen
@assert checkdiff_inferred(rosen, δrosen, randn(3))

# check basic use
@δ f(x, y) = (x + y) * y
@test checkdiff(f, δf, 2., 3.)

# check numerical constants
@δ function f2(x, y::AbstractFloat)
    srand(1)
    z = 4.0*3.2*2.5x - y^2 + rand(size(x)) + randn(size(x))
    sum(z) / y + randn() + rand()
end
@test checkdiff_inferred(f2, δf2, rand(5), rand())
@test checkdiff_inferred(f2, δf2, rand(3,3), rand())

# test broadcast
@δ f3(x) = sum(abs.(x))
@test checkdiff_inferred(f3, δf3, rand(5)-0.5)

# test multi argument functions and reuse
@δ f4(x, y) =  x * y, x - y
@δ function f5(x)
    (a, b) = f4(x, x+3)
    a * x + 4b
end
@test checkdiff_inferred(f5, δf5, rand()-0.5)

# test external constants
const f6r = rand(5)
@δ f6(x) = sum(f6r .* x.^2)
@test checkdiff_inferred(f6, δf6, rand(5))

# test ...
@δ function f7(x...)
    a, b = x
    return a * b
end
@test checkdiff_inferred(f7, δf7, rand(2)...)

# test ...
const f8_const = rand(13)
@δ function f8(a, x)
    b, c, d, e, f, g, h, i, j, k, l, m, n = x
    a * b + c * d - e * f + g * h - i * j / k * l + m * n
end
@δ test_f8(x) = f8(x, f8_const)
@test checkdiff_inferred(test_f8, δtest_f8, rand())

# test matrix multiply
@δ f9(x, y) = sum(x * y)
@test checkdiff_inferred(f9, δf9, rand(3), rand(3)')
@test checkdiff_inferred(f9, δf9, rand(3)', rand(3))
#@test checkdiff_inferred(f9, δf9, rand(3)', rand(3, 3))
@test checkdiff_inferred(f9, δf9, rand(3, 3), rand(3))
@test checkdiff_inferred(f9, δf9, rand(3, 3), rand(3, 3))

# test sequence of plus and times
@δ f10(x, y, z) = (x + y + z) * x * y * z
@test checkdiff_inferred(f10, δf10, rand(), rand(), rand())

# test multiple return values
@δ f11(x, y) = (x*y, y/x)
@δ function f12(x, y)
    return x+y, y-x
end
@δ function f13(x, y)
    a, b = f11(y, y)
    c, d = f12(x, b)
    return a+b+c+d+x
end
@δ function f14(x)
    f13(x, 2.3) + x
end
@test checkdiff(f14, δf14, rand())

# check constants
@δ function f15(x, y::Float64)
    srand(1)
    z = x + rand()
    return z+y
end
@δ function f16(x)
    f15(x, 2.3) + x
end
@test checkdiff_inferred(f16, δf16, rand())

# test tuple constants
@δ function f17(x)
    z = x.^3 + x
    state_const = (z, x.^2)
    state_const, x.^3
end
@δ function test_f17(x)
    state_const, y = f17(x)
    sum(y)
end
@test checkdiff_inferred(test_f17, δtest_f17, rand())
@test checkdiff_inferred(test_f17, δtest_f17, rand(3))

# single return symbol
@δ function f18(x)
    y = x + x
    return y
end
@test checkdiff_inferred(f18, δf18, rand())

# single return symbol
@δ function f19(x)
    y = x + x
    y
end
@test checkdiff_inferred(f19, δf19, rand())

# reassignment inside
@δ function f20(x)
    x = x + x
    x = x * x
    x
end
@test checkdiff_inferred(f20, δf20, rand())

# test broadcast
@eval @δ f21(x, y) = mean(f13.(x, y))
@eval @test checkdiff_inferred(f21, δf21, rand(), rand())
@eval @test checkdiff_inferred(f21, δf21, rand(3), rand(3, 2))
@eval @test checkdiff_inferred(f21, δf21, rand(3, 2), rand(3))
@eval @test checkdiff_inferred(f21, δf21, rand(5), rand())
@eval @test checkdiff_inferred(f21, δf21, rand(3, 2), rand())
@eval @test checkdiff_inferred(f21, δf21, rand(), rand(5))
@eval @test checkdiff_inferred(f21, δf21, rand(), rand(3, 2))

# (scalar, scalar), (scalar, const), (const, scalar), (const, const)
for o in [:+, :-, :*, :/, :^, :min, :max]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x, y) = $o(x, y) + $o(1.0, 3.0)
    @eval @test checkdiff_inferred($t, $δt, rand(), rand())
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = $o(x, 2.)
    @eval @test checkdiff_inferred($t, $δt, rand())
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = $o(2., x)
    @eval @test checkdiff_inferred($t, $δt, rand())
end

# (scalar)
for o in [:abs, :sum, :sqrt, :exp, :log, :-, :sign, :log1p, :expm1,
          :sin, :cos, :tan, :sinh, :cosh, :tanh, :asin, :acos, :atan,
          :round, :floor, :ceil, :trunc, :mod2pi, :maximum, :minimum, :transpose,
          :erf, :erfc, :gamma, :lgamma]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = $o(x) + x
    @eval @test checkdiff_inferred($t, $δt, rand())
end

# (vector, vector), (matrix, matrix), (const, *), (*, const), (const, const)
# (vector, matrix), (matrix, vector), (vector, scalar), (matrix, scalar), (scalar, vector), (scalar, matrix)
for o in [:.+, :.-, :.*, :./, :.^]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x, y) = sum($o(x, y)+$o(1.0, 3.0))
    @eval @test checkdiff_inferred($t, $δt, rand(5), rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand(3, 2))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o(x, 3.))
    @eval @test checkdiff_inferred($t, $δt, rand())
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o(3., x))
    @eval @test checkdiff_inferred($t, $δt, rand())
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x, y) = sum($o(x, y))
    @eval @test checkdiff_inferred($t, $δt, rand(), rand())
    @eval @test checkdiff_inferred($t, $δt, rand(3), rand(3, 2))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand(3))
    @eval @test checkdiff_inferred($t, $δt, rand(5), rand())
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand())
    @eval @test checkdiff_inferred($t, $δt, rand(), rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(), rand(3, 2))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o(x, ones(1, 2)))
    @eval @test checkdiff_inferred($t, $δt, rand())
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))
    @eval @test checkdiff_inferred($t, $δt, rand(1, 1))
    @eval @test checkdiff_inferred($t, $δt, rand(5, 1))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o(ones(1, 2), x))
    @eval @test checkdiff_inferred($t, $δt, rand())
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))
    @eval @test checkdiff_inferred($t, $δt, rand(1, 1))
    @eval @test checkdiff_inferred($t, $δt, rand(5, 1))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x, y) = sum($o(x, y))
    @eval @test checkdiff_inferred($t, $δt, rand(1, 1), rand(1, 1))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 1), rand(3, 2))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand(3, 1))
    @eval @test checkdiff_inferred($t, $δt, rand(5, 1), rand(1, 1))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand(1, 1))
    @eval @test checkdiff_inferred($t, $δt, rand(1, 1), rand(5, 1))
    @eval @test checkdiff_inferred($t, $δt, rand(1, 1), rand(3, 2))
end

# (scalar), (vector), (matrix)
for o in [:abs, :sqrt, :exp, :log, :sign, :log1p, :expm1,
          :sin, :cos, :tan, :sinh, :cosh, :tanh, :asin, :acos, :atan,
          :round, :floor, :ceil, :trunc, :mod2pi,
          :erf, :erfc, :gamma, :lgamma]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o.(x) + x)
    @eval @test checkdiff_inferred($t, $δt, rand())
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))
end

# (vector, scalar), (matrix, scalar), (scalar, vector), (scalar, matrix), (const, *), (*, const)
for o in [:+, :-, :*]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x, y) = sum($o(x, y))
    @eval @test checkdiff_inferred($t, $δt, rand(5), rand())
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand())
    @eval @test checkdiff_inferred($t, $δt, rand(), rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(), rand(3, 2))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o(x, 4.))
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o(5., x))
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))
end

# (vector, scalar), (matrix, scalar), (*, const)
for o in [:/]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x, y) = sum($o(x, y))
    @eval @test checkdiff_inferred($t, $δt, rand(5), rand())
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand())

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o(x, 5.))
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))
end

for o in [:dot]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x, y) = $o(x, y)
    @eval @test checkdiff_inferred($t, $δt, rand(5), rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand(3, 2))
end

for o in [:transpose, :maximum, :minimum]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o(x))
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))
end

# (vector, vector), (matrix, matrix), (const, *), (*, const)
for o in [:min, :max]
    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x, y) = sum($o.(x, y))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2), rand(3, 2))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o.(x, 3.))
    @eval @test checkdiff_inferred($t, $δt, rand())
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))

    t = gensym(o)
    δt = Symbol("δ$t")
    @eval @δ $t(x) = sum($o.(3., x))
    @eval @test checkdiff_inferred($t, $δt, rand())
    @eval @test checkdiff_inferred($t, $δt, rand(5))
    @eval @test checkdiff_inferred($t, $δt, rand(3, 2))
end

# https://github.com/gaika/AutoDiffSource.jl/issues/15
@δ f_15(x, y) = (x + 2y) * y^2
v, ∇f_25 = δf_15(3,2)
@test v == 28.0 && ∇f_25() == (4.0, 36.0)
