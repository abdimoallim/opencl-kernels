import device
import struct
import pyopencl as cl


class NDArray:
  def __init__(self, data, dtype=None, shape=None, context=None, queue=None):
    if context is None:
      context = device.create_gpu_context()
    if queue is None:
      queue = cl.CommandQueue(context)

    self.context = context
    self.queue = queue

    if isinstance(data, (list, tuple)):
      data = self._flatten_nested(data)
      if shape is None:
        shape = self._infer_shape(data)
    elif hasattr(data, "__iter__") and not isinstance(data, (str, bytes)):
      data = list(data)
      if shape is None:
        shape = (len(data),)

    if dtype is None:
      dtype = self._infer_dtype(data)

    self.dtype = dtype
    self.shape = tuple(shape) if shape is not None else ()
    self.size = self._compute_size()
    self.ndim = len(self.shape)

    if isinstance(data, cl.Buffer):
      self.data = data
    else:
      self.data = self._create_buffer_from_data(data)

  def _flatten_nested(self, data):
    result = []
    for item in data:
      if isinstance(item, (list, tuple)):
        result.extend(self._flatten_nested(item))
      else:
        result.append(item)
    return result

  def _infer_shape(self, data):
    if not isinstance(data, (list, tuple)):
      return ()
    shape = [len(data)]
    if data and isinstance(data[0], (list, tuple)):
      inner_shape = self._infer_shape(data[0])
      shape.extend(inner_shape)
    return tuple(shape)

  def _infer_dtype(self, data):
    if not data:
      return "float64"

    sample = data[0] if isinstance(data, (list, tuple)) else data

    if isinstance(sample, bool):
      return "bool"
    elif isinstance(sample, int):
      return "int64"
    elif isinstance(sample, float):
      return "float64"
    elif isinstance(sample, complex):
      return "complex128"
    else:
      return "float64"

  def _compute_size(self):
    size = 1
    for dim in self.shape:
      size *= dim
    return size

  def _create_buffer_from_data(self, data):
    if not isinstance(data, (list, tuple)):
      data = [data]

    dtype_map = {
      "bool": ("?", 1),
      "int8": ("b", 1),
      "int16": ("h", 2),
      "int32": ("i", 4),
      "int64": ("q", 8),
      "uint8": ("B", 1),
      "uint16": ("H", 2),
      "uint32": ("I", 4),
      "uint64": ("Q", 8),
      "float32": ("f", 4),
      "float64": ("d", 8),
    }

    format_char, item_size = dtype_map.get(self.dtype, ("d", 8))

    packed_data = struct.pack(f"{len(data)}{format_char}", *data)

    return cl.Buffer(self.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=packed_data)
