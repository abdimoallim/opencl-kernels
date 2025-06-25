import device


def main():
  device.list_opencl_devices()
  device.create_gpu_context()


if __name__ == "__main__":
  main()
