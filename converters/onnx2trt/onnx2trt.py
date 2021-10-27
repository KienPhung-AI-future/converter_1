import os
import tensorrt as trt
import sys
import argparse
import onnxruntime
import onnx
from typing import Tuple, Union
from loguru import logger
from reshape_onnx import reshape

# Based on code from NVES_R's response at
# https://forums.developer.nvidia.com/t/segmentation-fault-when-creating-the-trt-builder-in-python-works-fine-with-trtexec/111376


TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)


def _build_engine_onnx(input_onnx: Union[str, bytes], force_fp16: bool = False, max_batch_size: int = 1,
                       max_workspace: int = 1024):
    """
    Builds TensorRT engine from provided ONNX file

    :param input_onnx: serialized ONNX model.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful.
    :param max_batch_size: Define maximum batch size supported by engine. If >1 creates optimization profile.
    :param max_workspace: Maximum builder warkspace in MB.
    :return: TensorRT engine
    """

    with trt.Builder(TRT_LOGGER) as builder, \
            builder.create_network(EXPLICIT_BATCH) as network, \
            builder.create_builder_config() as config, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:

        if force_fp16 is True:
            logger.info('Building TensorRT engine with FP16 support.')
            has_fp16 = builder.platform_has_fast_fp16
            if not has_fp16:
                logger.warning('Builder report no fast FP16 support. Performance drop expected')
            config.set_flag(trt.BuilderFlag.FP16)
            config.set_flag(trt.BuilderFlag.STRICT_TYPES)

        config.max_workspace_size = max_workspace * 1024 * 1024

        if not parser.parse(input_onnx):
            logger.eror('ERROR: Failed to parse the ONNX')
            for error in range(parser.num_errors):
                print(parser.get_error(error))
            sys.exit(1)


        if max_batch_size != 1:
            loguru.warning('Batch size !=1 is used. Ensure your inference code supports it.')
        profile = builder.create_optimization_profile()
        # Get input name and shape for building optimization profile
        input = network.get_input(0)
        im_size = input.shape[2:]
        input_name = input.name
        profile.set_shape(input_name, (1, 3) + im_size, (1, 3) + im_size, (max_batch_size, 3) + im_size)
        config.add_optimization_profile(profile)

        return builder.build_engine(network, config=config)


def convert_onnx(input_onnx: Union[str, bytes], engine_file_path: str, force_fp16: bool = False,
                 max_batch_size: int = 1,):
    '''
    Creates TensorRT engine and serializes it to disk
    :param input_onnx: Path to ONNX file on disk or serialized ONNX model.
    :param engine_file_path: Path where TensorRT engine should be saved.
    :param force_fp16: Force use of FP16 precision, even if device doesn't support it. Be careful.
    :param max_batch_size: Define maximum batch size supported by engine. If >1 creates optimization profile.
    :return: None
    '''

    onnx_obj = None
    if isinstance(input_onnx, str):
        with open(input_onnx, "rb") as f:
            onnx_obj = f.read()
    elif isinstance(input_onnx, bytes):
        onnx_obj = input_onnx

    engine = _build_engine_onnx(input_onnx=onnx_obj,
                                force_fp16=force_fp16, max_batch_size=max_batch_size)

    assert not isinstance(engine, type(None))

    with open(engine_file_path, "wb") as f:
        f.write(engine.serialize())

    logger.success("Convert model successfully !!!")


def main():
    parser = argparse.ArgumentParser(description="Onnx to Tensorrt converter tool")
    parser.add_argument("-i", "--input_onnx", type=str, required=True)
    parser.add_argument("-o", "--engine_path", type=str, required=True)
    parser.add_argument("--force_fp16", action='store_true')
    parser.add_argument("--max_batch_size", type=int, default=1)

    args = parser.parse_args()
    # TODO Write the option for forge reshape input and the batchsize
    model = onnxruntime.InferenceSession(args.input_onnx, None)
    input_shape = model.get_inputs()[0].shape

    # if args.max_batch_size != 1:
    model = onnx.load(args.input_onnx)
    #reshaped = reshape(model, n=args.max_batch_size, h=800, w=1199)
    #temp_onnx_model = reshaped.SerializeToString()

    # else:
        # temp_onnx_model = args.input_onnx

    logger.info(f"Build TRT engine for {args.input_onnx}")
    convert_onnx(model, engine_file_path=args.engine_path,
                 max_batch_size=args.max_batch_size, force_fp16=args.force_fp16)


if __name__ == "__main__":
    main()
