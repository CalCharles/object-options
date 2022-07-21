# samplers aggregated here
from Causal.Sampling.General.uniform import UniformSampler
from Causal.Sampling.General.hist import HistorySampler
from Causal.Sampling.General.centered import CenteredSampler
from Causal.Sampling.General.exist import ExistSampler
from Causal.Sampling.General.dummy import DummySampler
samplers = {"uni": UniformSampler, "hist": HistorySampler, 'cent': CenteredSampler, "exist": ExistSampler, "dummy": DummySampler}  