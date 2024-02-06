# Latex OCR Server

A protobuf-based service to generate latex equations from image files.

See the [`.proto` file](https://github.com/lucasvanmol/latex-ocr-server/blob/0.1.0/src/latex_ocr_server/protos/latex_ocr.proto) for the interface.

## Installation

```
pip install latex-ocr-server
```

## Usage

```
usage: python -m latex_ocr_server start [-h] [--port PORT] [-d] [--cache_dir CACHE_DIR] [--cpu]

optional arguments:
  -h, --help            show this help message and exit
  --port PORT
  -d, --download        download model if needed without asking for confirmation
  --cache_dir CACHE_DIR
                        path to model cache. Defaults to ~/.cache/huggingface
  --cpu                 use cpu, otherwise uses gpu if available
```

## GPU support

`pytorch` must be installed with CUDA support. See https://pytorch.org/get-started/locally/.

You can check if gpu support is working with 
```
latex_ocr_server info --gpu-available
```

# Development 

## Build

```
hatch build
```

## Locall Install

```
pip install ./dist/latex_ocr_server-0.1.0.tar.gz
```
