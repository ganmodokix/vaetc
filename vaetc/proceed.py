from typing import Optional
import argparse

from vaetc.utils import debug_print
from .checkpoint import Checkpoint, load_checkpoint
from .train import train

def proceed(checkpoint: Checkpoint, extend_epochs: Optional[int] = None):
    """ Proceed the learning

    Args:
        checkpoint (Checkpoint): A training checkpoint to proceed
        extend_epochs: The extended number of epochs

    Returns:
        checkpoint (Checkpoint)
    """

    if extend_epochs is not None:
        debug_print(f"Learning extended to {extend_epochs} epochs ...")
        checkpoint.options["epochs"] = int(extend_epochs)
        checkpoint.options["until_convergence"] = False

    train(checkpoint)

    return checkpoint

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint_path",
        type=str, default="runs/current/checkpoint_last.pth",
        help="path to the checkpoint to load")
    parser.add_argument("--extend_epochs",
        type=int, default=None,
        help="extend learning if specified (not recommended)")

    args = parser.parse_args()
    
    debug_print(f"Loading {args.checkpoint_path} ...")
    checkpoint = load_checkpoint(args.checkpoint_path)

    proceed(checkpoint, extend_epochs=args.extend_epochs)
