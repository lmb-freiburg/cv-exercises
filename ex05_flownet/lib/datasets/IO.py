import re
import sys

import numpy as np
import imageio


def write(file, data):
    if file.endswith(".float3"):
        return writeFloat(file, data)
    elif file.endswith(".float4"):
        return writeFloat(file, data)
    elif file.endswith(".flo"):
        return writeFlow(file, data)
    elif file.endswith(".ppm"):
        return writeImage(file, data)
    elif file.endswith(".pgm"):
        return writeImage(file, data)
    elif file.endswith(".png"):
        return writeImage(file, data)
    elif file.endswith(".jpg"):
        return writeImage(file, data)
    elif file.endswith(".pfm"):
        return writePFM(file, data)
    else:
        raise Exception(f"Unknown file type: {file}")


def read(path):
    if path.endswith(".float3"):
        data = readFloat(path)
    elif path.endswith(".float4"):
        data = readFloat(path)
    elif path.endswith(".flo"):
        data = readFlow(path)
    elif path.endswith(".ppm"):
        data = readImage(path)
    elif path.endswith(".pgm"):
        data = readImage(path)
    elif path.endswith(".png"):
        data = readImage(path)
    elif path.endswith(".jpg"):
        data = readImage(path)
    elif path.endswith(".pfm"):
        data = readPFM(path)[0]
    else:
        raise Exception(f"Unknown file type: {path}")

    return np.asarray(data)


def writePFM(file, image, scale=1):
    file = open(file, "wb")

    color = None

    if image.dtype.name != "float32":
        raise Exception("Image dtype must be float32.")

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif (
        len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1
    ):  # greyscale
        color = False
    else:
        raise Exception("Image must have H x W x 3, H x W x 1 or H x W dimensions.")

    file.write("PF\n" if color else "Pf\n".encode())
    file.write("%d %d\n".encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == "<" or endian == "=" and sys.byteorder == "little":
        scale = -scale

    file.write("%f\n".encode() % scale)

    image.tofile(file)


def readPFM(file):
    file = open(file, "rb")

    header = file.readline().rstrip()
    if header.decode("ascii") == "PF":
        color = True
    elif header.decode("ascii") == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("ascii"))
    if dim_match:
        width, height = list(map(int, dim_match.groups()))
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().decode("ascii").rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    return data, scale


def writeFlow(name, flow):
    flow = np.squeeze(flow)
    if flow.shape[0] == 2:
        flow = flow.transpose(1, 2, 0)  # H, W, 2

    f = open(name, "wb")
    f.write("PIEH".encode("utf-8"))
    np.array([flow.shape[1], flow.shape[0]], dtype=np.int32).tofile(f)
    flow = flow.astype(np.float32)
    flow.tofile(f)


def readFlow(name):
    if name.endswith(".pfm") or name.endswith(".PFM"):
        return readPFM(name)[0][:, :, 0:2]

    f = open(name, "rb")

    header = f.read(4)
    if header.decode("utf-8") != "PIEH":
        raise Exception("Flow file header does not contain PIEH")

    width = np.fromfile(f, np.int32, 1).squeeze()
    height = np.fromfile(f, np.int32, 1).squeeze()

    flow = np.fromfile(f, np.float32, width * height * 2).reshape((height, width, 2))
    flow = flow.transpose(2, 0, 1)  # 2, H, W

    return flow.astype(np.float32)


def writeImage(name, data):
    if name.endswith(".pfm") or name.endswith(".PFM"):
        return writePFM(name, data, 1)

    return imageio.imwrite(name, data)


def readImage(name):
    if name.endswith(".pfm") or name.endswith(".PFM"):
        data = readPFM(name)[0]
        if len(data.shape) == 3:
            return data[:, :, 0:3]
        else:
            return data

    data = imageio.imread(name)

    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
        return data[0:3]
    else:
        return data


def writeFloat(name, data):  # expects NxCxHxW or CxHxW or HxW or W data
    f = open(name, "wb")
    dim = len(data.shape)

    f.write(("float\n").encode("ascii"))
    f.write(("%d\n" % dim).encode("ascii"))

    for i in range(dim):
        f.write(("%d\n" % data.shape[-1 - i]).encode("ascii"))

    data = data.astype(np.float32)
    data.tofile(f)


def readFloat(name):
    f = open(name, "rb")

    if (f.readline().decode("utf-8")) != "float\n":
        raise Exception("float file %s did not contain <float> keyword" % name)

    dim = int(f.readline())

    dims = []
    count = 1
    for i in range(0, dim):
        d = int(f.readline())
        dims.append(d)
        count *= d

    dims = list(reversed(dims))
    data = np.fromfile(f, np.float32, count).reshape(dims)
    return data  # Hxw or CxHxW NxCxHxW
