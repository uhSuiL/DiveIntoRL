def lazy_init(cls):
	class Tmp:
		def __init__(self, *args, **kwargs):
			self.args = args
			self.kwargs = kwargs

		def __call__(self):
			return cls(*self.args, **self.kwargs)

	return Tmp
