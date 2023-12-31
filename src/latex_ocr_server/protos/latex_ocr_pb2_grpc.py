# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

from . import latex_ocr_pb2 as latex__ocr__pb2


class LatexOCRStub(object):
    """Interface exported by the server
    """

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GenerateLatex = channel.unary_unary(
                '/latexocr.LatexOCR/GenerateLatex',
                request_serializer=latex__ocr__pb2.LatexRequest.SerializeToString,
                response_deserializer=latex__ocr__pb2.LatexReply.FromString,
                )
        self.IsReady = channel.unary_unary(
                '/latexocr.LatexOCR/IsReady',
                request_serializer=latex__ocr__pb2.Empty.SerializeToString,
                response_deserializer=latex__ocr__pb2.ServerIsReadyReply.FromString,
                )
        self.GetConfig = channel.unary_unary(
                '/latexocr.LatexOCR/GetConfig',
                request_serializer=latex__ocr__pb2.Empty.SerializeToString,
                response_deserializer=latex__ocr__pb2.ServerConfig.FromString,
                )


class LatexOCRServicer(object):
    """Interface exported by the server
    """

    def GenerateLatex(self, request, context):
        """Generate the latex code for a given image filepath
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def IsReady(self, request, context):
        """Check if the server is ready to return requests
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def GetConfig(self, request, context):
        """Get the server config
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_LatexOCRServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GenerateLatex': grpc.unary_unary_rpc_method_handler(
                    servicer.GenerateLatex,
                    request_deserializer=latex__ocr__pb2.LatexRequest.FromString,
                    response_serializer=latex__ocr__pb2.LatexReply.SerializeToString,
            ),
            'IsReady': grpc.unary_unary_rpc_method_handler(
                    servicer.IsReady,
                    request_deserializer=latex__ocr__pb2.Empty.FromString,
                    response_serializer=latex__ocr__pb2.ServerIsReadyReply.SerializeToString,
            ),
            'GetConfig': grpc.unary_unary_rpc_method_handler(
                    servicer.GetConfig,
                    request_deserializer=latex__ocr__pb2.Empty.FromString,
                    response_serializer=latex__ocr__pb2.ServerConfig.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'latexocr.LatexOCR', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class LatexOCR(object):
    """Interface exported by the server
    """

    @staticmethod
    def GenerateLatex(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/latexocr.LatexOCR/GenerateLatex',
            latex__ocr__pb2.LatexRequest.SerializeToString,
            latex__ocr__pb2.LatexReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def IsReady(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/latexocr.LatexOCR/IsReady',
            latex__ocr__pb2.Empty.SerializeToString,
            latex__ocr__pb2.ServerIsReadyReply.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def GetConfig(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/latexocr.LatexOCR/GetConfig',
            latex__ocr__pb2.Empty.SerializeToString,
            latex__ocr__pb2.ServerConfig.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
