import argparse
import logging
from huggingface_hub import try_to_load_from_cache, hf_hub_url
import urllib
from .__about__ import __version__

MODEL_NAME = "Norm/nougat-latex-base"

# Filename to check cache for
MODEL_FILE = "model.safetensors"

def run():
    logging.basicConfig()
    args = parse_option()
    args.func(args)

# latex_ocr_server info ...
def handle_info(args):
    if args.gpu_available:
        import torch

        is_available = torch.cuda.is_available()
        print(f"{is_available}")
        exit(not is_available)

# latex_ocr_server start ...
def handle_start(args):
    filepath = try_to_load_from_cache(MODEL_NAME, MODEL_FILE, cache_dir=args.cache_dir)
    if not isinstance(filepath, str) and not args.download:
        # Get file size
        download_path = hf_hub_url(MODEL_NAME, MODEL_FILE)
        req = urllib.request.Request(download_path, method="HEAD")
        f = urllib.request.urlopen(req)
        size = f.headers['Content-Length']
        file_size = '{:.2f} MB'.format(int(size) / float(1 << 20))

        # Get cache_dir name
        cache_dir = args.cache_dir if args.cache_dir else "~/.cache/huggingface"
        ans = input(f"Will download model ({file_size}) to {cache_dir}. Ok? (Y/n) ")
        if ans.lower() == "n":
            quit(0)
            
    from .server import serve
    serve(MODEL_NAME, args.port, args.cache_dir, args.cpu)
        
def parse_option():
    parser = argparse.ArgumentParser(prog="latex_ocr_server", description="A server that translates paths to images of equations to latex using protocol buffers.")
    parser.add_argument("--version", action='version', version=f'%(prog)s {__version__}')
    
    subparsers = parser.add_subparsers(help='sub-command help', required=True)
    start = subparsers.add_parser("start", help='start the server')
    start.add_argument("--port", default="50051")
    start.add_argument("-d", "--download", default=False, help="download model if needed without asking for confirmation", action='store_true')
    start.add_argument("--cache_dir", default=None, help="path to model cache. Defaults to ~/.cache/huggingface")
    start.add_argument("--cpu", default=False, action="store_true", help="use cpu, otherwise uses gpu if available")
    start.set_defaults(func=handle_start)

    info = subparsers.add_parser("info", help="get server info")
    info.add_argument("--gpu-available", required=True, action="store_true", help="check if gpu support is enabled")    
    info.set_defaults(func=handle_info)

    return parser.parse_args()

if __name__ == "__main__":
    run()