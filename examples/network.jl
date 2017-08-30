using ANN
xtrn, ytrn, xtst, ytst = ANN.loadmnist()

net = ANN.MlpClassifier()
ANN.preprocess!(net, xtrn, ytrn; splitratio = 0.2)

net = ANN.CnnClassifier()
ANN.preprocess!(net, xtrn, ytrn; shapes = (28, 28, 1), splitratio = 0.2)

ANN.hyperoptimize!(net; epochs = 20, maxevals = 300, random = true)
ANN.hyperoptimize!(net; epochs = 20, maxevals = 300, optim = true)
ANN.hyperoptimize!(net; epochs = 20, maxevals = 300, optim = false)
