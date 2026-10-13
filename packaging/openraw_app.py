"""Entry point for the frozen desktop app (PyInstaller). Kept tiny: the
multiprocessing guard must run before anything else in a frozen build."""

import multiprocessing
import sys

if __name__ == "__main__":
    multiprocessing.freeze_support()
    from openraw.gui.__main__ import main
    sys.exit(main())
