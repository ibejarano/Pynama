import yaml

def readYaml(yamlFile: str) -> dict:
    with open(f'{yamlFile}.yaml') as f:
        config = yaml.load(f, Loader=yaml.Loader)

    return config