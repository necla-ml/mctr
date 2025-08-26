import argparse
import re
from ml.utils import Config
from ml.argparse import ConfigAction

class ArgumentParser(argparse.ArgumentParser):
    r"""Allow no '=' in an argument config as a regular command line
    """
    def __init__(self, *args, **kwargs):
        char = kwargs.get("fromfile_prefix_chars", '@')
        kwargs["fromfile_prefix_chars"] = char
        super(ArgumentParser, self).__init__(*args, **kwargs)

    def convert_arg_line_to_args(self, line):
        tokens = line.strip().split()
        first = re.split(r"=", tokens[0])
        return first + tokens[1:]

    def parse_args(self, args=None, ns=None):
        args = super(ArgumentParser, self).parse_args(args, ns)
        cfg = Config(args)
        return cfg

    def parse_known_args(self, args=None, ns=None):
        args, leftover = super(ArgumentParser, self).parse_known_args(args, ns)
        cfg = Config(args)
        del cfg.CFG
        return cfg, leftover

def parse_args():
    parser = ArgumentParser()

    parser.add_argument('CMD', choices=['train', 'eval', 'extract', 'infer'], help='task to run')
    parser.add_argument('-s','--SEED', default='1204', type=int, help='Random seed')
    parser.add_argument("--CFG", action=ConfigAction, nargs='+', help="path to a configuration file in YAML/JSON or a Python module")

    # General options
    parser.add_argument('--DRYRUN', action='store_true', help='Run one iteration and skip the rest')
    parser.add_argument('--PREFIX', default='', type=str, help='prefix of the saved checkpoint filename')
    parser.add_argument('--RESUME', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
    parser.add_argument('--INITIALIZE', default='', type=str, metavar='PATH', help='path to checkpoint to initialize the model from(default: none)')
    # Logging & Tensorboard
    parser.add_argument('--LOGDIR', default='runs/train', help='tensorboard logdir')
    parser.add_argument('--LOG-PREFIX', default='exp', help='experiment prefix in tensorboard logdir')
    parser.add_argument('--NOTENSORBOARD', action='store_true', help='Disable logging to tensorboard')

    # DDP
    parser.add_argument('--WORLD-SIZE', default=-1, type=int, 
                        help='number of nodes for distributed training')
    parser.add_argument('--RANK', default=-1, type=int, 
                        help='node rank for distributed training')
    parser.add_argument('--DIST-URL', default='env://', type=str, 
                        help='url used to set up distributed training')
    parser.add_argument('--DIST-BACKEND', default='nccl', type=str, 
                        help='distributed backend')
    parser.add_argument('--DIST_NO_SYNC_BN', default=False, type=bool, 
                        help='disable batch norm sync')

    return parser.parse_args()