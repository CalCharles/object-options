# samplers aggregated here
from Causal.Sampling.General.uniform import UniformSampler
from Causal.Sampling.General.hist import HistorySampler
from Causal.Sampling.General.centered import CenteredSampler
samplers = {"uni": UniformSampler, "hist": HistorySampler, 'cent': CenteredSampler}  