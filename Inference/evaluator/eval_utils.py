from evaluator.comet_evaluator import CometEvaluator
from evaluator.dummy_evaluator import DummyEvaluator
from evaluator.d_bleu_evaluator import d_BleuEvaluator
#import evaluate


def get_eval_metrics(eval_metric_name):
    if "offline" in eval_metric_name:
        metric = DummyEvaluator()
    elif eval_metric_name == "comet":
        metric = CometEvaluator()
    elif eval_metric_name == "d-BLEU":
        #metric = evaluate.load(eval_metric_name)
        metric = d_BleuEvaluator()
    else:
        raise ValueError(f"Unknown eval metric {eval_metric_name}")

    return metric

