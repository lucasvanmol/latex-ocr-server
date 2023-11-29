from latex_ocr_server.protos import latex_ocr_pb2_grpc
from latex_ocr_server.protos import latex_ocr_pb2
import grpc
import logging

def run():
    with grpc.insecure_channel('localhost:50051') as channel:
        stub = latex_ocr_pb2_grpc.LatexOCRStub(channel)
        response = stub.GenerateLatex(latex_ocr_pb2.LatexRequest(image_path='./test.png'))
        print("Client received: " + response.latex)


if __name__ == "__main__":
    logging.basicConfig()
    run()
