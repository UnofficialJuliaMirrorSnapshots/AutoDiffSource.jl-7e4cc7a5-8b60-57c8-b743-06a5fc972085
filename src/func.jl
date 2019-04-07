const δceil_const = true
const δcolon_const = true
const δfloor_const = true
const δlength_const = true
const δones_const = true
const δrand_const = true
const δrandn_const = true
const δref_const2 = true
const δround_const = true
const δsign_const = true
const δsize_const = true
const δsrand_const = true
const δtrunc_const = true
const δtuple = true
const δzeros_const = true

function fit{T}(::Type{T}, x, sz::Tuple{Int, Int})::T
    if size(x) == sz
        x
    elseif sz[1] == 1 && sz[2] != 1
        sum(x, 1)
    elseif sz[2] == 1 && sz[1] != 1
        sum(x, 2)
    else
        fill(sum(x), sz)
    end
end
function fit{T}(::Type{T}, x, sz::Tuple{Int})::T
    if size(x) == sz
        vec(x)
    elseif sz[1] == 1
        fill(sum(x), sz)
    else
        vec(sum(x, 2))
    end
end
fit{T}(::Type{T}, x, sz::Tuple{})::T = sum(x)

safediv{T}(x::T, y) = y == 0 ? T(0) : x / y
δabs(x) = (abs(x), z->z*sign(x))
δacos(x) = (acos(x), z->-z/sqrt(1-x*x))
δasin(x) = (asin(x), z->z/sqrt(1-x*x))
δatan(x) = (atan(x), z->z/(1+x*x))
δcos(x) = (cos(x), z->-z*sin(x))
δcosh(x) = (cosh(x), z->z*sinh(x))
δdivide(x::AbstractArray, y::Real) = (t = x/y; (t, z->(z/y, -sum(z.*t)/y)))
δdivide(x::Real, y::Real) = (t = x/y; (t, z->(z/y, -z*t/y)))
δdivide_const1(x, y) = (t = x/y; (t, z->(-z*t/y)))
δdivide_const2(x, y) = (x/y, z->(z/y))
δdot(x, y) = (dot(x, y), z->(z.*y, z.*x))
δdot_divide_const1{T}(x, y::T) = (sy = size(y); t = x./y; (t, z->-fit(T, z.*t./y, sy)))
δdot_divide_const2{T}(x::T, y) = (sx = size(x); (x./y, z->fit(T, z./y, sx)))
δdot_divide{TX,TY}(x::TX, y::TY) = (sx = size(x); sy = size(y); t = x./y; (t, z->(fit(TX, z./y, sx), -fit(TY, z.*t./y, sy))))
δdot_minus_const1{T}(x, y::T) = (sy = size(y); (x.-y, z->-fit(T, z, sy)))
δdot_minus_const2{T}(x::T, y) = (sx = size(x); (x.-y, z->fit(T, z, sx)))
δdot_minus{TX,TY}(x::TX, y::TY) = (sx = size(x); sy = size(y); (x.-y, z->(fit(TX, z, sx), -fit(TY, z, sy))))
δdot_plus_const1{T}(x, y::T) = (sy = size(y); (x.+y, z->fit(T, z, sy)))
δdot_plus_const2{T}(x::T, y) = (sx = size(x); (x.+y, z->fit(T, z, sx)))
δdot_plus{TX,TY}(x::TX, y::TY) = (sx = size(x); sy = size(y); (x.+y, z->(fit(TX, z, sx), fit(TY, z, sy))))
δdot_power_const1{T}(x, y::T) = (sy = size(y); t = x.^y; (t, z->fit(T, z.*t.*log.(x), sy)))
δdot_power_const2{T}(x::T, y) = (sx = size(x); t = x.^y; (t, z->fit(T, y == 2 ? z.*2x : safediv.(z.*y.*t, x), sx)))
function δdot_power{TX,TY}(x::TX, y::TY)
    sx = size(x); sy = size(y)
    t = x.^y
    t, z->(fit(TX, safediv.(z.*y.*t, x), sx), fit(TY, z.*t.*log.(x), sy))
end
δdot_times_const1{T}(x, y::T) = (sy = size(y); (x.*y, z->fit(T, z.*x, sy)))
δdot_times_const2{T}(x::T, y) = (sx = size(x); (x.*y, z->fit(T, z.*y, sx)))
δdot_times{TX,TY}(x::TX, y::TY) = (sx = size(x); sy = size(y); (x.*y, z->(fit(TX, z.*y, sx), fit(TY, z.*x, sy))))
δerf(x) = (erf(x), y->y*2/sqrt(π)*exp(-x*x))
δerfc(x) = (erfc(x), y->-y*2/sqrt(π)*exp(-x*x))
δexp(x) = (t = exp(x); (t, z->z*t))
δexpm1(x) = (t = expm1(x); (t, z->z*(1 + t)))
δfanout(x) = (x..., (z...) -> z)
δgamma(x) = (t=gamma(x); (t, y->y*polygamma(0,x)*t))
δlgamma(x) = (lgamma(x), y->y*polygamma(0,x))
δlog(x) = (log(x), z->z/x)
δlog1p(x) = (log1p(x), z->z/(1 + x))
δmax(x, y) = (max(x, y), z->(z*(x>y),z*(x<y)))
δmax_const1(x, y) = (max(x, y), z->z*(x<y))
δmax_const2(x, y) = (max(x, y), z->z*(x>y))
δmaximum(x) = (t=maximum(x); (t, y->(t.==x).*y))
δmin(x, y) = (min(x, y), z->(z*(x<y),z*(x>y)))
δmin_const1(x, y) = (min(x, y), z->z*(x>y))
δmin_const2(x, y) = (min(x, y), z->z*(x<y))
δminimum(x) = (t=minimum(x); (t, y->(t.==x).*y))
δminus(x) = (-x, z->-z)
δminus(x::AbstractArray, y::Real) = (x-y, z->(z, -sum(z)))
δminus(x::Real, y::AbstractArray) = (x-y, z->(sum(z), -z))
δminus_const1(x, y) = (x-y, z->-z)
δminus_const2(x, y) = (x-y, z->z)
δminus_const1(x, y::Real) = (x-y, z->-sum(z))
δminus_const2(x::Real, y) = (x-y, z->sum(z))
δminus{T}(x::T, y::T) = (x-y, z->(z, -z))
δmod2pi(x::Real) = (mod2pi(x), y->y)
δplus(x::AbstractArray, y::Real) = (x+y, z->(z, sum(z)))
δplus(x::Real, y::AbstractArray) = (x+y, z->(sum(z), z))
δplus_const1(x, y) = (x+y, z->z)
δplus_const2(x, y) = (x+y, z->z)
δplus_const1(x, y::Real) = (x+y, z->sum(z))
δplus_const2(x::Real, y) = (x+y, z->sum(z))
δplus{T}(x::T, y::T) = (x+y, z->(z, z))
δpower(x::Real, y::Real) = (t = x^y; (t, z->(safediv(z*y*t, x), z*t*log(x))))
δpower_const1(x, y) = (t = x^y; (t, z->z*t*log(x)))
δpower_const2(x, y) = (t = x^y; (t, z->safediv(z*y*t, x)))
δsin(x) = (sin(x), z->z*cos(x))
δsinh(x) = (sinh(x), z->z*cosh(x))
δsqrt(x) = (t = sqrt(x); (t, z->0.5*z/t))
δsum(x::AbstractArray) = (t = size(x); (sum(x), z->fill(z, t)))
δsum(x::Real) = (x, z->z)
δtan(x) = (t = tan(x); (t, z->z*(1+t*t)))
δtanh(x) = (t = tanh(x); (t, z->z*(1-t*t)))
δtimes(x::AbstractVector, y::AbstractMatrix) = (x*y, z->(vec(z*y'), x'*z))
δtimes(x::AbstractMatrix, y::AbstractVector) = (x*y, z->(z*y', vec(x'*z)))
δtimes(x::AbstractMatrix, y::AbstractMatrix) = (x*y, z->(z*y', x'*z))
δtimes(x::AbstractArray, y::Real) = (x*y, z->(z.*y, sum(z.*x)))
δtimes(x::Real, y::AbstractArray) = (x*y, z->(sum(z.*y), z.*x))
δtimes(x::Real, y::Real) = (x*y, z->(z*y, z*x))
δtimes_const1(x, y) = (x*y, z->(x'*z))
δtimes_const1(x, y::Real) = (x*y, z->sum(x.*z))
δtimes_const2(x, y) = (x*y, z->(z*y'))
δtimes_const2(x::Real, y) = (x*y, z->sum(z.*y))
δtranspose(x::AbstractVector) = (x', y->vec(y'))
δtranspose(x) = (x', y->y')
δzeros(x::AbstractArray) = zeros(x)
δzeros{T}(x::T)::T = 0.
δmean(x::Real) = (x, z->z)
δmean(x::AbstractArray) = (t = size(x); (mean(x), z->fill(z/prod(t), t)))
