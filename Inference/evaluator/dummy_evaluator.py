class DummyEvaluator(object):
    def __init__(self, **eval_configs):
        pass

    def compute(self, predictions, references, sources):
        res = {"score": -1, "scores": [-1] * len(predictions)}
        return res

