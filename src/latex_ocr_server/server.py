from .protos import latex_ocr_pb2_grpc
from .protos import latex_ocr_pb2
import grpc
from PIL import Image
from transformers import VisionEncoderDecoderModel
from transformers.models.nougat import NougatTokenizerFast
from .nougat_latex_ocr.nougat_latex.util import process_raw_latex_code
from .nougat_latex_ocr.nougat_latex import NougatLaTexProcessor
import argparse
import torch
from concurrent import futures
import logging
import threading
from huggingface_hub import try_to_load_from_cache, hf_hub_url
import urllib

MODEL_NAME = "Norm/nougat-latex-base"

# Filename to check cache for
MODEL_FILE = "model.safetensors"

def parse_option():
    parser = argparse.ArgumentParser(prog="latex_ocr_server", description="A server that translates paths to images of equations to latex using protocol buffers.")
    parser.add_argument("--version", action='store_true', help="display version and quit")
    
    subparsers = parser.add_subparsers(help='sub-command help')
    start = subparsers.add_parser("start", help='start the server')
    start.add_argument("--port", default="50051")
    start.add_argument("-d", "--download", default=False, help="download model if needed without asking for confirmation", action='store_true')
    start.add_argument("--cache_dir", default=None, help="path to model cache. Defaults to ~/.cache/huggingface")
    start.add_argument("--cpu", default=False, action="store_true", help="use cpu, otherwise uses gpu if available")

    info = subparsers.add_parser("info", help="get server info")
    info.add_argument("--gpu-available", required=True, action="store_true", help="check if gpu support is enabled")    

    return parser.parse_args()

class LatexOCR(latex_ocr_pb2_grpc.LatexOCRServicer):
    def __init__(self, model, cache_dir, device):
        self.device = device
        self.cache_dir = cache_dir
        self.model = None
        self.tokenizer = None
        self.latex_processor = None

        self.load_thread = threading.Thread(target=self.load_models, args=(model,), daemon=True)
        print("Server started")
        self.load_thread.start()

    def GenerateLatex(self, request, context):
        image = Image.open(request.image_path)
        if not image.mode == "RGB":
            image = image.convert('RGB')
        result = self.inference(image)
        return latex_ocr_pb2.LatexReply(latex=result)
    
    def IsReady(self, request, context):
        is_ready = not self.load_thread.is_alive()
        return latex_ocr_pb2.ServerIsReadyReply(is_ready=is_ready)
    
    def GetConfig(self, request, context):
        return latex_ocr_pb2.ServerConfig(device = self.device, cache_dir=self.cache_dir)
    
    def inference(self, image):
        pixel_values = self.latex_processor(image, return_tensors="pt").pixel_values
        task_prompt = self.tokenizer.bos_token
        decoder_input_ids = self.tokenizer(task_prompt, add_special_tokens=False,
                                    return_tensors="pt").input_ids
        with torch.no_grad():
            outputs = self.model.generate(
                pixel_values.to(self.device),
                decoder_input_ids=decoder_input_ids.to(self.device),
                max_length=self.model.decoder.config.max_length,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                bad_words_ids=[[self.tokenizer.unk_token_id]],
                return_dict_in_generate=True,
            )
        sequence = self.tokenizer.batch_decode(outputs.sequences)[0]
        sequence = sequence.replace(self.tokenizer.eos_token, "").replace(self.tokenizer.pad_token, "").replace(self.tokenizer.bos_token,"")
        return process_raw_latex_code(sequence)

    def load_models(self, model):
        print("Loading model...", end="")
        self.model = VisionEncoderDecoderModel.from_pretrained(model, cache_dir=self.cache_dir, resume_download=True).to(self.device)
        print(" done")

        print("Loading processor...", end="")
        self.tokenizer = NougatTokenizerFast.from_pretrained(model, cache_dir=self.cache_dir, resume_download=True)
        self.latex_processor = NougatLaTexProcessor.from_pretrained(model, cache_dir=self.cache_dir, resume_download=True)
        print(" done")


def serve(port: str, cache_dir: str, cpu: bool):
    if not cpu and torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
    print(f"Starting server on port {port}, using {device}")
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    latex_ocr_pb2_grpc.add_LatexOCRServicer_to_server(LatexOCR(MODEL_NAME, cache_dir, device), server)
    server.add_insecure_port("[::]:" + port)
    server.start()
    server.wait_for_termination()

def run():
    logging.basicConfig()
    args = parse_option()
    if args.version:
        from .__about__ import __version__
        print(f"latex_ocr_server {__version__}")
    elif hasattr(args, "gpu_available"):
        print(f"{torch.cuda.is_available()}")
    else:
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
        serve(args.port, args.cache_dir, args.cpu)

if __name__ == "__main__":
    run()