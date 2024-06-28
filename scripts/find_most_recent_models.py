from pathlib import Path

import omegaconf


if __name__ == "__main__":
    dir_list = sorted(list(Path('outputs').glob("2024-0?-??/*")), key=lambda x: x.stat().st_mtime, reverse=True)

    for path in dir_list:
        model_files = sorted(list(path.glob("*.tgz")), key=lambda x: len(str(x)))
        if model_files:
        cfg = path / ".hydra" / "config.yaml"
        config_dict = omegaconf.OmegaConf.load(cfg)
        print(config_dict.keys())
