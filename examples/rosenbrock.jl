using AutoDiffSource
using BenchmarkTools
using ReverseDiffSource

rosenbrock(x, y) = sum(100*(y-x.^2).^2 + (1.-x).^2)

@δ rosenbrock(x) = rosenbrock(x[1:length(x)-1], x[2:length(x)])

# verify correctness
@assert checkdiff(rosenbrock, δrosenbrock, randn(3))

# handcrafted derivative
function ∇rosenbrock(x)
    x1 = x[1:end-1]
    x2 = x[2:end]
    ∂x = similar(x)
    ∂x[1:end-1] = - 400*x1.*(x2-x1.^2) - 2*(1-x1)
    ∂x[end] = 0
    ∂x[2:end] += 200*(x2-x1.^2)
    ∂x
end

# verify that it is correct
const x0 = randn(3)
@assert checkgrad(rosenbrock, (x0,), ∇rosenbrock(x0))

rds_rosenbrock = rdiff(rosenbrock, (Vector{Float64}, Vector{Float64}))

# benchmark
const x1 = randn(1_000)
trial_manual = @benchmark (rosenbrock(x1), ∇rosenbrock(x1));
trial_rds = @benchmark ((y, ∂x1, ∂x2) = rds_rosenbrock(x1[1:end-1], x1[2:end]);
                        ∂x = similar(x1);
                        ∂x[1:end-1] = ∂x1;
                        ∂x[end] = 0;
                        ∂x[2:end] += ∂x2;
                        ∂x)
trial_auto = @benchmark ((y, ∇r) = δrosenbrock(x1); ∇r())

# auto should be ~15% faster than handcrafted and 60% faster than reverse diff source
@show trial_manual
@show trial_rds
@show trial_auto

""
