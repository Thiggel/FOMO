from experiment.__main__ import set_checkpoint_for_run
from argparse import Namespace


def test_set_checkpoint_for_run():
    args = Namespace(checkpoint=["path/to/checkpoint1", "path/to/checkpoint2"])
    args = set_checkpoint_for_run(args, 0)

    assert args.checkpoint == "path/to/checkpoint1"

    args = Namespace(checkpoint=["path/to/checkpoint1", "path/to/checkpoint2"])
    args = set_checkpoint_for_run(args, 1)

    assert args.checkpoint == "path/to/checkpoint2"

    args = Namespace(checkpoint=["path/to/checkpoint1", "path/to/checkpoint2"])
    args = set_checkpoint_for_run(args, 2)

    assert args.checkpoint == "path/to/checkpoint1"
