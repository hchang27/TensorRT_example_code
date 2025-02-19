"""Convert ONNX models to TensorRT engines.
Mostly scraped from https://github.com/onnx/onnx-tensorrt
Enhanced with the following features:
    - Support FP32/FP16/INT8 precision
    - Support serialization/deserialization of TensorRT engines to speed up
"""
from __future__ import print_function
import os
import tensorrt as trt
from onnx.backend.base import Backend, BackendRep, Device, DeviceType, namedtupledict
import onnx
import torch
import ctypes
from onnx import helper as onnx_helper
import numpy as np
import six
from six import string_types
import pycuda.driver
import pycuda.gpuarray
import pycuda.autoinit
import numba
from numba import cuda
# HACK Should look for a better way/place to do this
from ctypes import cdll, c_char_p
libcudart = cdll.LoadLibrary('libcudart.so')
libcudart.cudaGetErrorString.restype = c_char_p
TORCH_NP_DTYPE_MAP = {
    # signed integers
    torch.int8: np.int8,
    torch.int16: np.int16,
    torch.short: np.int16,
    torch.int32: np.int32,
    torch.int: np.int32,
    torch.int64: np.int64,
    torch.long: np.int64,
    # unsinged inters
    torch.uint8: np.uint8,
    # floating point
    torch.float: np.float32,
    torch.float32: np.float32,
    torch.float16: np.float16,
    torch.half: np.float16,
    torch.float64: np.float64,
    torch.double: np.float64
}
NP_TORCH_DTYPE_MAP = {v: k for k, v in TORCH_NP_DTYPE_MAP.items()}
def cudaSetDevice(device_idx):
    """Set the device to the given device index."""
    ret = libcudart.cudaSetDevice(device_idx)
    if ret != 0:
        error_string = libcudart.cudaGetErrorString(ret)
        if isinstance(error_string, bytes):
            error_string = error_string.decode("utf-8")
        raise RuntimeError("cudaSetDevice: " + error_string)
def count_trailing_ones(vals):
    """Count the number of trailing ones in a list of values."""
    count = 0
    for val in reversed(vals):
        if val != 1:
            return count
        count += 1
    return count
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)
class Binding(object):
    def __init__(self, engine, idx_or_name):
        if isinstance(idx_or_name, string_types):
            self.name = idx_or_name
        else:
            self.index = idx_or_name
            self.name  = engine.get_tensor_name(self.index)
            if self.name is None:
                raise IndexError("Binding index out of range: %i" % self.index)
        self.is_input = engine.get_tensor_mode(self.name) == trt.TensorIOMode.INPUT
        dtype = engine.get_tensor_dtype(self.name)
        dtype_map = {trt.DataType.FLOAT:  np.float32,
                        trt.DataType.HALF:  np.float16,
                        trt.DataType.INT8:  np.int8,
                        trt.DataType.BOOL:  np.bool_,}
        if hasattr(trt.DataType, 'INT32'):
            dtype_map[trt.DataType.INT32] = np.int32
        if hasattr(trt.DataType, 'INT64'):
            dtype_map[trt.DataType.INT64] = np.int64
        self.dtype = dtype_map[dtype]
        shape = engine.get_tensor_shape(self.name)
        self.shape = tuple(shape)
        self._host_buf   = None
        self._device_buf = None
    @property
    def host_buffer(self):
        if self._host_buf is None:
            self._host_buf = pycuda.driver.pagelocked_empty(self.shape, self.dtype)
        return self._host_buf
    
    @property
    def device_buffer(self):
        if self._device_buf is None:
            self._device_buf = pycuda.gpuarray.empty(self.shape, self.dtype)
        return self._device_buf
    
    def get_async(self, stream, input_output_mode):
        src = self.device_buffer
        if input_output_mode == 'numpy':
            dst = self.host_buffer
            src.get_async(stream, dst)
        elif input_output_mode == 'torch_cuda':
            # Failed attempts: ctypes pointer conversion. Direct as_tensor. dtod memcpy (issue seems to be pytorch memory).
            output_arr = cuda.as_cuda_array(src)  # uses __cuda_array_interface__
            output_arr = torch.as_tensor(output_arr, device='cuda')
            return output_arr
        else:
            raise ValueError("Invalid input_output_mode: %s" % input_output_mode)
        return dst
def squeeze_hw(x):
    if x.shape[-2:] == (1, 1):
        x = x.reshape(x.shape[:-2])
    elif x.shape[-1] == 1:
        x = x.reshape(x.shape[:-1])
    return x
def check_input_validity(input_idx, input_array, input_binding, input_output_mode):
    # Check shape
    trt_shape = tuple(input_binding.shape)
    onnx_shape    = tuple(input_array.shape)
    gpu_ptr_copy_flag = False
    if onnx_shape != trt_shape:
        if not (trt_shape == (1,) and onnx_shape == ()) :
            raise ValueError("Wrong shape for input %i. Expected %s, got %s." %
                            (input_idx, trt_shape, onnx_shape))
    
    if input_output_mode == 'torch_cuda':
        assert input_array.is_cuda
        if TORCH_NP_DTYPE_MAP[input_array.dtype] == input_binding.dtype:
            # subsequent logic performs ptr-to-ptr copy
            gpu_ptr_copy_flag = True
        else:
            raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                            (input_idx, input_binding.dtype, input_array.dtype))
    elif input_output_mode == 'numpy':
        # Check dtype
        if input_array.dtype != input_binding.dtype:
            #TRT does not support INT64, need to convert to INT32
            if input_array.dtype == np.int64 and input_binding.dtype == np.int32:
                casted_input_array = np.array(input_array, copy=True, dtype=np.int32)
                if np.equal(input_array, casted_input_array).all():
                    input_array = casted_input_array
                else:
                    raise TypeError("Wrong dtype for input %i. Expected %s, got %s. Cannot safely cast." %
                                (input_idx, input_binding.dtype, input_array.dtype))
            else:
                raise TypeError("Wrong dtype for input %i. Expected %s, got %s." %
                                (input_idx, input_binding.dtype, input_array.dtype))
    else:
        raise ValueError("Invalid input_output_mode: %s" % input_output_mode)
    return input_array, gpu_ptr_copy_flag
class Engine(object):
    def __init__(self, trt_engine):
        self.engine = trt_engine
        bindings = [Binding(self.engine, i)
                    for i in range(self.engine.num_io_tensors)]
        self.binding_addrs = [b.device_buffer.ptr for b in bindings]
        self.inputs  = [b for b in bindings if     b.is_input]
        self.outputs = [b for b in bindings if not b.is_input]
        
        for binding in self.inputs + self.outputs:
            _ = binding.device_buffer # Force buffer allocation
        for binding in self.outputs:
            _ = binding.host_buffer   # Force buffer allocation
        self.context = self.engine.create_execution_context()
        self.stream = pycuda.driver.Stream()
    def __del__(self):
        if self.engine is not None:
            del self.engine
    def run(self, inputs, input_output_mode):
        assert input_output_mode in ['torch_cuda', 'numpy']
        # len(inputs) > len(self.inputs) with Shape operator, input is never used
        # len(inputs) == len(self.inputs) for other operators
        if len(inputs) < len(self.inputs):
            raise ValueError("Not enough inputs. Expected %i, got %i." %
                             (len(self.inputs), len(inputs)))
        if isinstance(inputs, dict):
            inputs = [inputs[b.name] for b in self.inputs]
        for i, (input_array, input_binding) in enumerate(zip(inputs, self.inputs)):
            input_array, gpu_ptr_copy_flag = check_input_validity(i, input_array, input_binding, input_output_mode)
            input_binding_array = input_binding.device_buffer
            if gpu_ptr_copy_flag:
                pycuda.driver.memcpy_dtod_async(input_binding_array.ptr, input_array.data_ptr(), input_binding_array.nbytes, self.stream)
                # this raises illegal memory access error in internal TRT engine.
                # input_binding_array.gpudata = input_array.data_ptr()
            else:
                input_binding_array.set_async(input_array, self.stream)
        num_io = self.engine.num_io_tensors
        for i in range(num_io):
            tensor_name = self.engine.get_tensor_name(i)
            if i < len(inputs) and self.engine.is_shape_inference_io(tensor_name):
                self.context.set_tensor_address(tensor_name, inputs[i].ctypes.data)
            else:
                self.context.set_tensor_address(tensor_name, self.binding_addrs[i])
        self.context.execute_async_v3(self.stream.handle)
        results = [output.get_async(self.stream, input_output_mode) for output in self.outputs]
        self.stream.synchronize()
        return results
    def run_no_dma(self):
        self.context.execute_async_v3(self.stream.handle)
class TensorRTBackendRep(BackendRep):
    """Wrapper for TensorRT backend rep."""
    def __init__(self, model, device,
                 max_workspace_size=None, serialize_engine=False, verbose=False,
                 serialized_engine_path=None, int8_calibrator=None, **kwargs):
        """Initialize a TensorRT backend rep.
        Args:
            model (onnx.ModelProto): ONNX model
            device (Device): device to run inference on
            max_workspace_size (int, optional): maximum workspace size. Defaults to None.
            serialize_engine (bool, optional): whether to serialize the engine. Defaults to False.
            verbose (bool, optional): whether to print verbose information. Defaults to False.
            serialized_engine_path (str, optional): path to serialized engine. Defaults to None.
        """
        if not isinstance(device, Device):
            device = Device(device)
        self._set_device(device)
        self.serialized_engine_path = serialized_engine_path
        if self.serialized_engine_path is not None:
            assert serialize_engine
        self._logger = TRT_LOGGER
        # Fore more builder config options, see
        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/Builder.html
        self.builder = trt.Builder(self._logger)
        self.config = self.builder.create_builder_config()
        # For more config options, see
        # https://docs.nvidia.com/deeplearning/tensorrt/api/python_api/infer/Core/BuilderConfig.html
        self.int8_calibrator = int8_calibrator
        if self.builder.platform_has_fast_fp16:
            print("FAST FP16 detected. Enabling precision to FP16...")
            self.serialized_engine_path = self.serialized_engine_path.replace('.trt', '_fp16.trt')
            self.config.set_flag(trt.BuilderFlag.FP16)
        # TODO(roger): enable INT8 requires post-training quantization and calibration
        if self.builder.platform_has_fast_int8 and self.int8_calibrator is not None:
            print("FAST INT8 detected. Enabling INT8...")
            self.serialized_engine_path = self.serialized_engine_path.replace('.trt', '_int8.trt')
            self.config.set_flag(trt.BuilderFlag.INT8)
            self.config.int8_calibrator = self.int8_calibrator
            # TODO: where should this go?
            # self.config.set_calibration_profile(profile)
        self.network = self.builder.create_network(flags=1 << (
            int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
        self.parser = trt.OnnxParser(self.network, self._logger)
        self.shape_tensor_inputs = []
        self.serialize_engine = serialize_engine
        self.verbose = verbose
        if self.verbose:
            print(f'\nRunning {model.graph.name}...')
            TRT_LOGGER.min_severity = trt.Logger.VERBOSE
        if not isinstance(model, six.string_types):
            model_str = model.SerializeToString()
        else:
            model_str = model
        if not trt.init_libnvinfer_plugins(TRT_LOGGER, ""):
            msg = "Failed to initialize TensorRT's plugin library."
            raise RuntimeError(msg)
        if not self.parser.parse(model_str):
            error = self.parser.get_error(0)
            msg = "While parsing node number %i:\n" % error.node()
            msg += ("%s:%i In function %s:\n[%i] %s" %
                    (error.file(), error.line(), error.func(),
                     error.code(), error.desc()))
            raise RuntimeError(msg)
        if max_workspace_size is None:
            max_workspace_size = 1 << 28
        self.config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, max_workspace_size)
        if self.verbose:
            for layer in self.network:
                print(layer)
            print(f'Output shape: {self.network[-1].get_output(0).shape}')
        if os.path.exists(self.serialized_engine_path):
            del self.parser
            self.runtime = trt.Runtime(TRT_LOGGER)
            print("Loading serialized engine from {}".format(self.serialized_engine_path))
            with open(self.serialized_engine_path, 'rb') as f:
                trt_engine = self.runtime.deserialize_cuda_engine(f.read())
            self.engine = Engine(trt_engine)
        else:
            print("First time building engine. This may take a while... (up to 20 minutes)")
            self._build_engine()
        self._output_shapes = {}
        self._output_dtype = {}
        for output in model.graph.output:
            dims = output.type.tensor_type.shape.dim
            output_shape = tuple([dim.dim_value for dim in dims])
            self._output_shapes[output.name] = output_shape
            self._output_dtype[output.name] = output.type.tensor_type.elem_type
    def _build_engine(self, inputs=None):
        """Build a TensorRT engine with a builder config.
        Args:
            inputs(List of np.ndarray): inputs to the model; if not None,
                        this means we are building the engine at run time,
                        because we need to register optimization profiles for some inputs
        """
        opt_profile = None
        if inputs:
            opt_profile = self.builder.create_optimization_profile()
            # Set optimization profiles for the input bindings that need them
            for i in range(self.network.num_inputs):
                inp_tensor = self.network.get_input(i)
                name = inp_tensor.name
                # Set profiles for shape tensors
                if inp_tensor.is_shape_tensor:
                    if inputs[i].ndim > 0:
                        val_list = inputs[i].tolist()
                        opt_profile.set_shape_input(name, val_list, val_list, val_list)
                    else:
                        opt_profile.set_shape_input(name, [inputs[i]], [inputs[i]], [inputs[i]])
                # Set profiles for dynamic execution tensors
                elif -1 in inp_tensor.shape:
                    opt_profile.set_shape(name, inputs[i].shape, inputs[i].shape, inputs[i].shape)
            self.config.add_optimization_profile(opt_profile)
        if self.int8_calibrator is not None:
            if opt_profile is None:
                opt_profile = self.builder.create_optimization_profile()
            self.config.set_calibration_profile(opt_profile)
        trt_blob = self.builder.build_serialized_network(self.network, self.config)
        if trt_blob is None:
            raise RuntimeError("Failed to build TensorRT engine from network")
        
        trt_engine = self._deserialize(trt_blob)
        if self.serialize_engine:
            trt_engine = self._serialize_deserialize(trt_engine, self.serialized_engine_path)
        self.engine = Engine(trt_engine)
    def _set_device(self, device):
        """Set the device to the given device index.
        Args:
            device (Device): device to run inference on
        """
        self.device = device
        assert (device.type == DeviceType.CUDA)
        cudaSetDevice(device.device_id)
    
    def _deserialize(self, trt_blob):
        self.runtime = trt.Runtime(TRT_LOGGER)
        del self.parser # Parser no longer needed for ownership of plugins
        trt_engine = self.runtime.deserialize_cuda_engine(trt_blob)
        return trt_engine
    def _serialize_deserialize(self, trt_engine, serialized_engine_path):
        """Serialize and deserialize the engine to speed up future runs.
        Args:
            trt_engine (trt.ICudaEngine): TensorRT engine
            serialized_engine_path (str): path to serialized engine
        Returns:
            trt.ICudaEngine: TensorRT engine
        """
        # TODO(roger): unify load and save functions
        self.runtime = trt.Runtime(TRT_LOGGER)
        serialized_engine = trt_engine.serialize()
        if serialized_engine_path is not None:
            with open(serialized_engine_path, 'wb') as f:
                f.write(serialized_engine)
            print("Serialized engine written to {}".format(serialized_engine_path))
        trt_engine = self.runtime.deserialize_cuda_engine(
            serialized_engine)
        return trt_engine
    def run(self, inputs, input_output_mode='numpy', **kwargs):
        """Execute the prepared engine and return the outputs as a named tuple.
        Args:
            inputs -- Input tensor(s) as a Numpy array or list of Numpy arrays.
            input_output_mode (str, optional): input_output_mode (str, optional): 'numpy' or 'torch_cuda'. Defaults to 'numpy'.
        """
        if isinstance(inputs, np.ndarray) or isinstance(inputs, torch.Tensor):
            inputs = [inputs]
        outputs = self.engine.run(inputs, input_output_mode)
        output_names = [output.name for output in self.engine.outputs]
        for i, (name, array) in enumerate(zip(output_names, outputs)):
            output_shape = self._output_shapes[name]
            # HACK WAR for unknown output shape in run_node
            if output_shape == (-99,):
                # WAR for TRT requiring at least 2 dims (NC)
                min_dims = 2
                if _tensorrt_version()[0] < 4:
                    # WAR for TRT only supporting 4D (NCHW) tensors
                    min_dims = 4
                if array.ndim == min_dims:
                    npadding_dims = count_trailing_ones(array.shape)
                    if npadding_dims > 0:
                        outputs[i] = array.reshape(
                            array.shape[:-npadding_dims])
            else:
                # HACK WAR replace fixed batch dim with variable
                if self._output_dtype[name] == onnx.TensorProto.INT64:
                    if array.dtype == np.int32:
                        casted_output = np.array(outputs[i], dtype=np.int64)
                        if np.equal(outputs[i], casted_output).all():
                            outputs[i] = np.array(outputs[i], dtype=np.int64)
                if self._output_dtype[name] == onnx.TensorProto.DOUBLE:
                    if array.dtype == np.float32:
                        casted_output = np.array(outputs[i], dtype=np.double)
                        if np.equal(outputs[i], casted_output).all():
                            outputs[i] = np.array(outputs[i], dtype=np.double)
        outputs_tuple = namedtupledict('Outputs', output_names)(*outputs)
        return namedtupledict('Outputs', output_names)(*outputs)
class TensorRTBackend(Backend):
    """TensorRT backend. Wrapper around ONNX backend."""
    @classmethod
    def prepare(cls, onnx_model_path, device='CUDA:0', **kwargs):
        """Build an engine from the given model.
        Args:
            onnx_model_path (str): path to ONNX model
            device (str, optional): device to run inference on. Defaults to 'CUDA:0'.
        """
        model = onnx.load(onnx_model_path)
        super(TensorRTBackend, cls).prepare(model, device, **kwargs)
        return TensorRTBackendRep(model, device, **kwargs)