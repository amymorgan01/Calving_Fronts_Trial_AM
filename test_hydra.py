import hydra
from omegaconf import DictConfig

@hydra.main(version_base=None, config_path="conf", config_name="config")  #TODO: could change to train and test
def main(cfg: DictConfig):
    print(cfg)
    print("all going swimingly")


if __name__ == "__main__":
    main()