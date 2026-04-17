import hydra
from omegaconf import DictConfig


@hydra.main(version_base=None, config_path=".", config_name="config")
def cc_processor(config: DictConfig) -> None:
    print(config, type(config))


if __name__ == "__main__":
    cc_processor()
