import argparse

import os
import datetime
import subprocess
from vaetc.utils import debug_print
import yaml

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--python", "-p", type=str, default="python", help="python command")
    parser.add_argument("--dry_run", "-d", action="store_true", help="dry run")
    parser.add_argument("--runs_dir", "-r", type=str, default="runs", help="path of runs")
    args = parser.parse_args()

    for entry in os.scandir(args.runs_dir):

        if entry.is_dir():

            # debug_print(f"Scannig {entry.path} ...")
            
            need_evaluation = False
            
            status_path = os.path.join(entry.path, "exit_status.yaml")
            if os.path.isfile(status_path):
                
                mtime = os.stat(status_path).st_mtime
                mtime = datetime.datetime.fromtimestamp(mtime)
                if (mtime.year, mtime.month, mtime.day) < (2021, 12, 1):
                    need_evaluation = True

            se_path = os.path.join(entry.path, "metrics_test_4.yaml")
            if not os.path.exists(se_path):
                need_evaluation = True

            if need_evaluation:

                debug_print(f"Redo needed for {entry.path}")
                
                if not args.dry_run:
                    subprocess.run([args.python, "eval.py", "--logger_path", entry.path])
            else:

                pass
                # debug_print(f"Skipped")