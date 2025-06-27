import pyopencl as cl


def list_opencl_devices():
  for platform in cl.get_platforms():
    print(f"Platform: {platform.name}")
    print(f"Vendor: {platform.vendor}")
    print(f"Version: {platform.version}")

    for device in platform.get_devices():
      print(f"  Device: {device.name}")
      print(f"    Type: {cl.device_type.to_string(device.type)}")
      print(f"    Max compute units: {device.max_compute_units}")
      print(f"    Max work group size: {device.max_work_group_size}")
      print(f"    Global memory: {device.global_mem_size // (1024**2)} MB")
      print(f"    Local memory: {device.local_mem_size // 1024} KB")


def create_gpu_context():
  """Create context for GPU with CPU fallback"""
  try:
    context = cl.Context(dev_type=cl.device_type.GPU)
    return context
  except:  # noqa: E722
    return cl.Context(dev_type=cl.device_type.CPU)
