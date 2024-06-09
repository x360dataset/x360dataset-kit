import yaml, argparse
import os, re
import git
from .utils import Dict2Class

def read_yaml(file):
    with open(file) as f:
        data = yaml.safe_load(f)
    data = Dict2Class(data)
    return data

def get_git_root(path):
    git_repo = git.Repo(path, search_parent_directories=True)
    git_root = git_repo.git.rev_parse("--show-toplevel")
    return git_root

def get_env():
    cfg = read_yaml(os.path.join(get_git_root("."), "./configs/env.yaml"))
    cfg.REPO_ROOT = get_git_root(".")
    return cfg
