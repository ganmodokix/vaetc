import os
import sys
import argparse

sys.path.append(os.path.dirname(__file__) + '/../')
import vaetc

def main(checkpoint_path: str):
    
    checkpoint = vaetc.load_checkpoint(checkpoint_path)
    vaetc.evaluate(checkpoint)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint_path", type=str, help="checkpoint_path")

    main(parser.parse_args().checkpoint_path)