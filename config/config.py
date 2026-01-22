import yaml

with open("../config.yaml", 'r') as stream:
    cfg = yaml.load(stream, Loader=yaml.SafeLoader)

