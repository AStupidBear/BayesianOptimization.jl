module BayesianOptimization

using Distributions, GaussianProcesses, BlackBoxOptim, Logging
export rosenbrock, branin, branin_slow
include("util.jl")

logstream = open(joinpath(tempdir(), "BayesOpt.log"), "a")
Logging.configure(output = [logstream, STDOUT], level = Logging.DEBUG)

srand(100)

type BayesOpt
    f::Function
    X::Array{Float64, 2}
    y::Array{Float64, 1}
    xmax::Array{Float64, 1}
    ymax::Float64
    encoder::BoundEncoder
    model
end

function optimize!(opt::BayesOpt, maxevals = 1, optim = false, random = false)
    np = nprocs()
    i = last_index = length(opt.y) + 1
    # extend opt.X and opt.y to store new data points
    opt.X = hcat(opt.X, zeros(size(opt.X, 1), maxevals))
    append!(opt.y, fill(-1e10, maxevals))
    # pmap like asyncronous update
    nextidx() = (idx=i; i+=1; idx)
    @sync for p = 1:np
        if p != myid() || np == 1
            @async while true
                idx = nextidx()
                idx > last_index + maxevals - 1 && break
                # sample and transform a new data point
                x_new = (idx < np || random) ?
                        [rand(b) for b in opt.encoder.bounds] :
                        acquire_max(opt.model, opt.encoder.bounds)
                x_new = purturb(x_new, opt.X[:, 1:idx], opt.encoder.bounds)
                c_new = transform(opt.encoder, x_new)
                y_new = remotecall_fetch(opt.f, p, c_new)
                # store the new data point and update the model
                opt.X[:, idx], opt.y[idx] = x_new, y_new
                yscale = centralize(opt.y[1:idx])
                opt.model = GP(opt.X[:, 1:idx], yscale, MeanConst(mean(yscale)), SE(0.0, 0.0), -5.0)
                optim && try GaussianProcesses.optimize!(opt.model) end
                y_new > opt.ymax && ((opt.xmax, opt.ymax) = (x_new, y_new))
                # debug & report
                debug("\niteration = $idx, new x = $x_new, y = $y_new")
                debug("\niteration = $idx, x_max = $(opt.xmax), ymax = $(opt.ymax)")
                report(opt)
            end
        end
    end
    info("\nOptimization completed:\n xmax = $(opt.xmax), ymax = $(opt.ymax)")
    return opt.xmax, opt.ymax
end

function acquire_max(model, bounds)
    ymax = maximum(model.y)
    opt = bbsetup(expected_improvement(model, ymax); SearchRange = bounds, TraceMode = :silent)
    res = bboptimize(opt)
    x_max = best_candidate(res)
end

function expected_improvement(model, ymax)
    function ei(x)
        μ, Σ = GaussianProcesses.predict_y(model, colvec(x))
        Σ == 0 && return 0
        σ = sqrt(Σ); Z = (μ - ymax) / σ
        res = -((μ - ymax) * cdf(Normal(), Z) + σ * pdf(Normal(), Z))[1]
    end
    return ei
end

function optimize(f::Function, bounds, c0 = []; maxevals = 100, optim = false, random = false, o...)
    encoder = BoundEncoder(bounds)
    X, y = zeros(length(encoder.bounds), 1), zeros(1)
    # evaluate on the initial point
    xmax = isempty(c0) ? [rand(b) for b in encoder.bounds] : inverse_transform(encoder, c0)
    cmax = transform(encoder, xmax)
    maxevals == 1 && return cmax, -1e10, [-1e10]
    ymax = f(cmax)
    X[:, 1], y[1] = xmax, ymax
    # sample the rest points
    opt = BayesOpt(f, X, y, xmax, ymax, encoder, GP(X[:, 1:1], [0.0], MeanZero(), SE(0.0, 0.0)))
    optimize!(opt, maxevals - 1, optim, random)
    cmax = transform(opt.encoder, opt.xmax)
    return cmax, opt.ymax, progress(opt)
end

progress(opt::BayesOpt) = maximums(opt.y)

maximize(f, args...; kwargs...) = optimize(f, args...; kwargs...)

function minimize(f, args...; kwargs...)
    cmax, ymax, prog = optimize(x -> -f(x), args...; kwargs...)
    cmin, ymin, prog = cmax, -ymax, -prog
end

function report(opt::BayesOpt)
    if isdefined(Main, :Plots)
      Main.plot(progress(opt))
      Main.savefig(joinpath(tempdir(), "BayesOpt.html"))
    end
end

end
