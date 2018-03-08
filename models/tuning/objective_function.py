class ObjectiveFunction:
    """class to analyze objective function optimization : hyperparameter tuning"""
    def __init__(self, func):
        self.f = func
        self.history_f = []
        self.history_fbest = None
        self.history_bests = []

    def __call__(self, x):
        val = self.f(x)
        self.history_f.append(-val)
        if self.history_fbest is None:
            self.history_fbest = val
            self.history_bests.append(-val)
        elif self.history_fbest > val:
            self.history_fbest = val
            self.history_bests.append(-val)
        else:
            self.history_bests.append(-self.history_fbest)
        return val
