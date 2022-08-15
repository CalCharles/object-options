from Causal.dummy_interaction import DummyInteraction
from Causal.Dummy.dummy_all import DummyAllInteraction
from Causal.Dummy.dummy_laser import DummyLaserInteraction
from Causal.Dummy.dummy_target import DummyTargetInteraction

dummy_interactions = {'base': DummyInteraction, 'laser': DummyLaserInteraction, 'all': DummyAllInteraction, "target": DummyTargetInteraction}