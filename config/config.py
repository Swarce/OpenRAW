import argparse

BATCH_SIZE = 1

DATA_PATH = "./data/"



def get_arguments():
    parser = argparse.ArgumentParser(description="training codes")
    
    parser.add_argument("--task", type=str, help="Name of this training")
    parser.add_argument("--data_path", type=str, default=DATA_PATH, help="Dataset root path.")
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size for training. ")       
    parser.add_argument("--debug_mode", dest='debug_mode', action='store_true',  help="If debug mode, load less data.")    
    parser.add_argument("--gamma", dest='gamma', action='store_true', help="Use gamma compression for raw data.")     
    # PATCHED (OpenRAW, not upstream): any FiveK camera, not just upstream's two.
    # Repeatable / comma-separated; several cameras are pooled into one model.
    # Accepts FiveK names ("Nikon D70") or prepared folder names ("NIKON_D700").
    parser.add_argument("--camera", action="append", default=None,
                        help="FiveK camera(s) to train on, e.g. 'Nikon D700' (repeatable). Default: NIKON_D700.")
    parser.add_argument("--list-cameras", dest="list_cameras", action="store_true",
                        help="list FiveK cameras with image counts and exit")
    parser.add_argument("--all-downloaded", dest="all_downloaded", action="store_true",
                        help="train on every camera found in data/fivek/raw/, using whatever is downloaded")
    parser.add_argument("--download", action="store_true",
                        help="download any missing DNGs for the chosen camera(s) before training")
    parser.add_argument("--prepare-only", dest="prepare_only", action="store_true",
                        help="download/preprocess the data, then exit (no GPU needed)")
    parser.add_argument("--download-jobs", dest="download_jobs", type=int, default=4, help="parallel downloads")
    parser.add_argument("--rgb_weight", type=float, default=1, help="Weight for rgb loss. ")                 
    
    
    return parser
