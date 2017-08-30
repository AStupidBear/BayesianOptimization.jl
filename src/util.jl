colvec(x) = reshape(x, length(x), 1)

function centralize!(x, dim=1)
  _max = maximum(x, dim)
  _min = minimum(x, dim)
  x .= (x .- _min) ./ (_max .- _min)
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
            a+=asinh(i+j)
        end
    end
    return a
end

function rosenbrock(x)
  x = collect(x)
  z = sum( 100*( x[2:end] .- x[1:end-1].^2 ).^2 .+ ( x[1:end-1] .- 1 ).^2 )
  Float64(z)
end

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

discretize(c::Range, x) = c[indmin(abs2(x .- c))]

discretize(c::NTuple{2, Number}, x) = x < c[1] ? c[1] : x > c[2] ? c[2] : x

discretize(c::Vector, x) = (i = discretize(1:length(c), x); c[i])


"""
    configs = ((-1, 1), 1, 1:10, ["1", 3, 2])
    encoder = BoundEncoder(configs)
    x = [0, 5.3, 3]
    c = transform(encoder, x)
    @assert x == inverse_transform(encoder, c)
"""
type BoundEncoder
  configs::Tuple
  bounds::Array{NTuple{2, Float64}}
end

function BoundEncoder(configs)
  bounds = []
  for c in configs length(c) > 1 && push!(bounds, tobound(c)) end
  configs = tuple(configs...)
  bounds = [Float64.(extrema(b)) for b in bounds]
  BoundEncoder(configs, bounds)
end

function transform(encoder::BoundEncoder, x)
  i, c = 0, []
  for cc in encoder.configs
    push!(c, length(cc) > 1 ? (i += 1; discretize(cc, x[i])) : cc)
  end
  c
end

function inverse_transform(encoder::BoundEncoder, c)
  x = Float64[c[i] for i in eachindex(c) if length(encoder.configs[i]) > 1]
end
