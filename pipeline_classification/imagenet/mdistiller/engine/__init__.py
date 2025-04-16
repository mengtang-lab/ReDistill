from .trainer import BaseTrainer, CRDTrainer, AugTrainer, ReedTrainer

trainer_dict = {
    "base": BaseTrainer,
    "crd": CRDTrainer,
    "mlld": AugTrainer,
    "reed": ReedTrainer,
}
