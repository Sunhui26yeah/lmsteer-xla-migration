import random
import sys
import torch
import numpy as np


def get_device(args):
    if getattr(args, "xla", False):
        import torch_xla
        import torch_xla.backends
        # match CUDA fp32 matmuls (TF32 off by default) instead of the TPU's
        # reduced-precision default for f32
        torch_xla.backends.set_mat_mul_precision("highest")
        return torch_xla.device()
    return torch.device("cuda:0") if args.cuda else torch.device("cpu")


def xla_metrics_line(tag=""):
    import torch_xla.debug.metrics as met

    def count(name):
        data = met.metric_data(name)
        return data[0] if data else 0
    aten = {k: met.counter_value(k) for k in met.counter_names()
            if k.startswith("aten::")}
    return (f"XLA_METRICS {tag} compile={count('CompileTime')} "
            f"execute={count('ExecuteTime')} aten_fallbacks={aten}")


def set_seed(seed):
    if seed is None:
        return
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    if "torch_xla" in sys.modules:
        import torch_xla.core.xla_model as xm
        xm.set_rng_state(seed)


class RunningMean:
    def __init__(self, gamma):
        self.gamma = gamma
        self.count = 0
        self._value = None

    def update(self, value):
        value = value.detach().cpu()
        if value.ndim == 0:
            self._update(value)
        else:
            for _v in value:
                self._update(_v)

    def _update(self, value):
        self.count += 1
        if self._value is None:
            self._value = value
        else:
            w1 = self.gamma * (1 - self.gamma ** (self.count - 1))
            w2 = (1 - self.gamma)
            wt = w1 + w2
            w1 = w1 / wt
            w2 = w2 / wt
            self._value = w1 * self._value + w2 * value

    @property
    def value(self):
        if self._value is None:
            return 0
        return self._value * 1
