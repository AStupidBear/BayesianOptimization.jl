using BayesianOptimization; reload("BayesianOptimization")
BO = BayesianOptimization

x0, bounds = rand((0, 1), 2), [(0, 1), (0, 1)]
BO.minimize(branin, bounds, x0; optim = true, maxevals = 10)
BO.minimize(branin, bounds, x0; optim = false, maxevals = 10)
BO.minimize(branin, bounds, x0; random = true, maxevals = 10)
BO.cmaminimize(branin,  bounds, x0; σ0 = 0.2, maxevals = 10)
nprocs() > 1 && BO.minimize(branin_slow, bounds, x0; optim = true, maxevals = 10)

x0, bounds = rand((0, 2), 10), [(0, 2) for i in 1:10]
BO.minimize(rosenbrock, bounds, x0; optim = true, maxevals = 30)
BO.minimize(rosenbrock, bounds, x0; optim = false, maxevals = 30)
BO.minimize(rosenbrock, bounds, x0; random = true, maxevals = 30)
BO.cmaminimize(rosenbrock, bounds, x0; σ0 = 0.2, maxevals = 30)

function benchmark(f, bounds, x0; maxevals = 20, restarts = 10)
  loss = [zeros(maxevals, restarts) for i in 1:4]
  for t in 1:restarts
    loss[1][:, t] = BO.minimize(f, bounds, x0; maxevals = maxevals, optim = false)[3]
    loss[2][:, t] = BO.minimize(f, bounds, x0; maxevals = maxevals, optim = true)[3]
    loss[3][:, t] = BO.minimize(f, bounds, x0; maxevals = maxevals, random = true)[3]
    loss[4][:, t] = BO.cmaminimize(f, bounds, x0; maxevals = maxevals)[2]
  end
  loss = map(x -> median(x, 2), loss)
  plot(loss, label = ["no_opt" "opt" "random" "cmaes"])
  savefig("bayesopt_$(f)_benchmark.html")
end

using Plots; plotly()

x0, bounds = rand((0, 1), 2), [(0, 1), (0, 1)]
benchmark(branin, bounds, x0)

x0, bounds = rand((0, 2), 10), [(0, 2) for i in 1:10]
benchmark(rosenbrock, bounds, x0)
