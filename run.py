"""Launch skill-based RL training."""

import hydra
from omegaconf import OmegaConf, DictConfig

from rolf.main import Run


class SkillRLRun(Run):
    def _set_run_name(self):
        """Sets run name."""
        cfg = self._cfg
        if "phase" in cfg.rolf:
            cfg.run_name = f"{cfg.env.id}.{cfg.rolf.name}.{cfg.rolf.phase}.{cfg.run_prefix}.{cfg.seed}"
        else:
            super()._set_run_name()

    def _get_trainer(self):
        if self._cfg.rolf.name in ["spirl_dreamer", "spirl_tdmpc", "skimo"]:
            from skill_trainer import SkillTrainer

            return SkillTrainer(self._cfg)
        if self._cfg.rolf.name == "spirl":
            from spirl_trainer import SPiRLTrainer

            return SPiRLTrainer(self._cfg)
        return super()._get_trainer()


@hydra.main(config_path="config", config_name="default_config")
def main(cfg: DictConfig) -> None:
    # Make config writable
    OmegaConf.set_struct(cfg, False)

    # Change default config
    cfg.wandb_entity = "clvr"
    cfg.wandb_project = "p-skill-model"

    # Execute training code
    SkillRLRun(cfg).run()


if __name__ == "__main__":
    main()
