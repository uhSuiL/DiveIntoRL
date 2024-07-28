from torch import nn


def lazy_init(cls):
	class Tmp:
		def __init__(self, *args, **kwargs):
			self.args = args
			self.kwargs = kwargs

		def __call__(self):
			return cls(*self.args, **self.kwargs)

	return Tmp


def MLP(layers_dims: list[int]) -> nn.Module:
	layers = []
	for i in range(len(layers_dims) - 2):
		layers += [nn.Linear(layers_dims[i], layers_dims[i+1]), nn.ReLU()]
	layers.append(nn.Linear(layers_dims[-2], layers_dims[-1]))
	return nn.Sequential(*layers)