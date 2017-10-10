colvec(x) = reshape(x, length(x), 1)

function centralize!(x, dim=1)
  𝑚ax = maximum(x, dim)
  𝑚in = minimum(x, dim)
  x .= (x .- 𝑚in) ./ (𝑚ax .- 𝑚in)
  x .= 2 .* x .- 1
end

centralize(x, dim = 1) = centralize!(deepcopy(x), dim)

function maximums(x)
  maxs = similar(x)
  xmax = x[1]
  @inbounds for i = eachindex(x)
    x[i] > xmax && (xmax = x[i])
    maxs[i] = xmax
  end
  maxs
end

function slow()
    a = 1.0
    for i in 1:10000
        for j in 1:10000
            a += asinh(i + j)
        end
    end
    return a
end

function rosenbrock(x)
  x = collect(x)
  z = sum( 100*( x[2:end] .- x[1:end-1].^2 ).^2 .+ ( x[1:end-1] .- 1 ).^2 )
  Float64(z)
end

rastrigin(x) = 10length(x) + sum(x.^2 - 10cos(2π * x))

# fmin = -1.04
function branin(v)
    x, y = v
    x, y = 15x - 5, 15y
    res = 1/51.95 * ((y - 5.1*x^2 / (4*π^2) + 5x/π - 6)^2 + (10 -10/8π)cos(x) -44.81)
end

branin_slow(v) = (slow(); branin(v))

Base.rand(b::NTuple{2, Real}, dims...) = b[1] + rand(dims...) * (b[2] - b[1])

function purturb(x, X, bounds)
  for j in 1:size(X, 2)
    if isapprox(x, X[:, j])
      for i in 1:length(x)
        x[i] += rand(bounds[i])
      end
    end
  end
  x
end

tobound(b) = b

tobound(b::Vector) = 1:length(b)

discretize(cfg::Range, x) = cfg[indmin(abs2(x .- cfg))]

inverse_discretize(cfg::Range, c) = c

discretize(cfg::NTuple{2, Number}, x) = x < cfg[1] ? cfg[1] : x > cfg[2] ? cfg[2] : x

inverse_discretize(cfg::NTuple{2, Number}, c) = c

discretize(cfg::Vector, x) = (i = discretize(1:length(cfg), x); cfg[i])

inverse_discretize(cfg::Vector, c) = findfirst(x -> x == c, cfg)

length2(x::String) = 1

length2(x) = length(x)

"""
    using BayesianOptimization; BO = BayesianOptimization
    configs = ((-1, 1), 1, 1:10, ["1", 3, 2])
    encoder = BO.BoundEncoder(configs)
    y = [0, 7, 1]
    x = [2 * (yi - b[1]) / (b[2] - b[1]) - 1 for (yi, b) in zip(y, encoder.orgbounds)]
    c = BO.transform(encoder, x)
    @assert x == BO.inverse_transform(encoder, c)
"""
type BoundEncoder
  configs::Tuple
  orgbounds::Array{NTuple{2, Float64}}
  bounds::Array{NTuple{2, Float64}}
end

function BoundEncoder(configs)
  bounds = []
  for cfg in configs length2(cfg) > 1 && push!(bounds, tobound(cfg)) end
  configs = tuple(configs...)
  orgbounds = [Float64.(extrema(b)) for b in bounds]
  BoundEncoder(configs, orgbounds, [(-0.5, 0.5) for b in bounds])
end

function transform(encoder::BoundEncoder, x)
  i, c = 0, []
  for cfg in encoder.configs
    if length2(cfg) > 1
        i += 1
        b = encoder.orgbounds[i]
        y = (x[i] + 1) * (b[2] - b[1]) / 2 + b[1]
        push!(c, discretize(cfg, y))
    else
        push!(c,  cfg)
    end
  end
  return c
end

function inverse_transform(encoder::BoundEncoder, c)
  y = [inverse_discretize(cfg, ci) for (cfg, ci) in zip(encoder.configs, c) if length2(cfg) > 1]
  x = [2 * (yi - b[1]) / (b[2] - b[1]) - 1 for (yi, b) in zip(y, encoder.orgbounds)]
end


# type BoundEncoder
#   configs::Tuple
#   bounds::Array{NTuple{2, Float64}}
# end
#
# function BoundEncoder(configs)
#   bounds = []
#   for cfg in configs length2(cfg) > 1 && push!(bounds, tobound(cfg)) end
#   configs = tuple(configs...)
#   bounds = [Float64.(extrema(b)) for b in bounds]
#   BoundEncoder(configs, bounds)
# end
#
# function transform(encoder::BoundEncoder, x)
#   i, c = 0, []
#   for cfg in encoder.configs
#     push!(c, length2(cfg) > 1 ? (i += 1; discretize(cfg, x[i])) : cfg)
#   end
#   c
# end
#
# function inverse_transform(encoder::BoundEncoder, c)
#   x = [inverse_discretize(cfg, cc) for (cfg, cc) in zip(encoder.configs, c) if length2(cfg) > 1]
# end
