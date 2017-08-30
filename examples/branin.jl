using BayesianOptimization
BO = BayesianOptimization

BO.minimize(branin, [(0, 1), (0, 1)], [0.5, 0.5]; optim = true, maxevals = 10)
BO.minimize(branin, [(0, 1), (0, 1)], [0.5, 0.5]; random = true, maxevals = 10)
nprocs() > 1 && BO.minimize(branin_slow, [(0, 1), (0, 1)]; optim = true, maxevals = 10)

function benchmark(f, bounds; maxevals = 10, restarts = 3)
  loss = [zeros(maxevals, restarts) for i in 1:3]
  for t in 1:restarts
    loss[1][:, t] = BO.minimize(f, bounds; maxevals = maxevals, optim = false)[3]
    loss[2][:, t] = BO.minimize(f, bounds; maxevals = maxevals, optim = true)[3]
    loss[3][:, t] = BO.minimize(f, bounds; maxevals = maxevals, random = true)[3]
  end
  loss = map(x -> median(x, 2), loss)
  plot(loss, label = ["no_opt" "opt" "random"])
  savefig(timename("bayesopt_$(f)_benchmark.html"))
end

using Plots; plotly()

benchmark(branin, [(0, 1), (0, 1)])

benchmark(rosenbrock, [(0, 2) for i in 1:10])
