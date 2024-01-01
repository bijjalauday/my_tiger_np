import contextlib
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import urllib
import warnings
import zipfile
from pathlib import Path
from urllib.parse import urlparse

from ta_pet_id.yolo.utils.general import check_suffix, yaml_load


def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = (
            nn.SiLU()
            if act is True
            else (act if isinstance(act, nn.Module) else nn.Identity())
        )

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(
        self, c1, c2, k=1, s=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x


class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(
            *(TransformerLayer(c2, num_heads) for _ in range(num_layers))
        )
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)


class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(
        self, c1, c2, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))


class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))


class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(
        self, c1, c2, n=1, shortcut=True, g=1, e=0.5
    ):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(
            *(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n))
        )
        # self.m = nn.Sequential(*[CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)])

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))


class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)


class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList(
            [nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k]
        )

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(
        self, c1, c2, k=1, s=1, p=None, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(
            torch.cat(
                [
                    x[..., ::2, ::2],
                    x[..., 1::2, ::2],
                    x[..., ::2, 1::2],
                    x[..., 1::2, 1::2],
                ],
                1,
            )
        )
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(
        self, c1, c2, k=1, s=1, g=1, act=True
    ):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)


class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(
            GhostConv(c1, c_, 1, 1),  # pw
            DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
            GhostConv(c_, c2, 1, 1, act=False),
        )  # pw-linear
        self.shortcut = (
            nn.Sequential(
                DWConv(c1, c1, k, s, act=False), Conv(c1, c2, 1, 1, act=False)
            )
            if s == 2
            else nn.Identity()
        )

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        (
            b,
            c,
            h,
            w,
        ) = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s**2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s**2, h * s, w * s)  # x(1,16,160,160)


class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
        return torch.cat(x, self.d)


# from yolo v5 git
def export_formats():
    # YOLOv5 export formats
    x = [
        ["PyTorch", "-", ".pt", True, True],
        ["TorchScript", "torchscript", ".torchscript", True, True],
        ["ONNX", "onnx", ".onnx", True, True],
        ["OpenVINO", "openvino", "_openvino_model", True, False],
        ["TensorRT", "engine", ".engine", False, True],
        ["CoreML", "coreml", ".mlmodel", True, False],
        ["TensorFlow SavedModel", "saved_model", "_saved_model", True, True],
        ["TensorFlow GraphDef", "pb", ".pb", True, True],
        ["TensorFlow Lite", "tflite", ".tflite", True, False],
        ["TensorFlow Edge TPU", "edgetpu", "_edgetpu.tflite", False, False],
        ["TensorFlow.js", "tfjs", "_web_model", False, False],
        ["PaddlePaddle", "paddle", "_paddle_model", True, True],
    ]
    return pd.DataFrame(x, columns=["Format", "Argument", "Suffix", "CPU", "GPU"])


def is_url(url, check=True):
    # Check if string is URL and check if URL exists
    try:
        url = str(url)
        result = urllib.parse.urlparse(url)
        assert all([result.scheme, result.netloc])  # check if is url
        return (
            (urllib.request.urlopen(url).getcode() == 200) if check else True
        )  # check if exists online
    except (AssertionError, urllib.request.HTTPError):
        return False


class DetectMultiBackend(nn.Module):
    # YOLOv5 MultiBackend class for python inference on various backends
    def __init__(
        self,
        weights="yolov5s.pt",
        device=torch.device("cpu"),
        dnn=False,
        data=None,
        fp16=False,
        fuse=True,
    ):
        # Usage:
        #   PyTorch:              weights = *.pt
        #   TorchScript:                    *.torchscript
        #   ONNX Runtime:                   *.onnx
        #   ONNX OpenCV DNN:                *.onnx --dnn
        #   OpenVINO:                       *_openvino_model
        #   CoreML:                         *.mlmodel
        #   TensorRT:                       *.engine
        #   TensorFlow SavedModel:          *_saved_model
        #   TensorFlow GraphDef:            *.pb
        #   TensorFlow Lite:                *.tflite
        #   TensorFlow Edge TPU:            *_edgetpu.tflite
        #   PaddlePaddle:                   *_paddle_model
        from models.experimental import (  # scoped to avoid circular import; attempt_download,
            attempt_load,
        )

        super().__init__()
        w = str(weights[0] if isinstance(weights, list) else weights)
        (
            pt,
            jit,
            onnx,
            xml,
            engine,
            coreml,
            saved_model,
            pb,
            tflite,
            edgetpu,
            tfjs,
            paddle,
            triton,
        ) = self._model_type(w)
        fp16 &= pt or jit or onnx or engine or triton  # FP16
        nhwc = (
            coreml or saved_model or pb or tflite or edgetpu
        )  # BHWC formats (vs torch BCWH)
        stride = 64  # default stride
        cuda = torch.cuda.is_available() and device.type != "cpu"  # use CUDA
        # if not (pt or triton):
        #     w = attempt_download(w)  # download if not local

        if pt:  # PyTorch
            model = attempt_load(
                weights if isinstance(weights, list) else w,
                map_location=device,
                inplace=True,
                fuse=fuse,
            )
            stride = max(int(model.stride.max()), 32)  # model stride
            names = (
                model.module.names if hasattr(model, "module") else model.names
            )  # get class names
            model.half() if fp16 else model.float()
            self.model = model  # explicitly assign for to(), cpu(), cuda(), half()


        elif (
            tflite or edgetpu
        ):  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
            try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
                from tflite_runtime.interpreter import (
                    Interpreter,
                    load_delegate,
                )
            except ImportError:
                import tensorflow as tf

                Interpreter, load_delegate = (
                    tf.lite.Interpreter,
                    tf.lite.experimental.load_delegate,
                )
            if edgetpu:  # TF Edge TPU https://coral.ai/software/#edgetpu-runtime
                # LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
                delegate = {
                    "Linux": "libedgetpu.so.1",
                    "Darwin": "libedgetpu.1.dylib",
                    "Windows": "edgetpu.dll",
                }[platform.system()]
                interpreter = Interpreter(
                    model_path=w, experimental_delegates=[load_delegate(delegate)]
                )
            else:  # TFLite
                # LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
                interpreter = Interpreter(model_path=w)  # load TFLite model
            interpreter.allocate_tensors()  # allocate
            input_details = interpreter.get_input_details()  # inputs
            output_details = interpreter.get_output_details()  # outputs
            # load metadata
            with contextlib.suppress(zipfile.BadZipFile):
                with zipfile.ZipFile(w, "r") as model:
                    meta_file = model.namelist()[0]
                    meta = ast.literal_eval(model.read(meta_file).decode("utf-8"))
                    stride, names = int(meta["stride"]), meta["names"]

        else:
            raise NotImplementedError(f"ERROR: {w} is not a supported format")

        # class names
        if "names" not in locals():
            names = (
                yaml_load(data)["names"]
                if data
                else {i: f"class{i}" for i in range(999)}
            )
        if names[0] == "n01440764" and len(names) == 1000:  # ImageNet
            names = yaml_load(ROOT / "data/ImageNet.yaml")[
                "names"
            ]  # human-readable names

        self.__dict__.update(locals())  # assign all variables to self

    def forward(self, im, augment=False, visualize=False):
        # YOLOv5 MultiBackend inference
        b, ch, h, w = im.shape  # batch, channel, height, width
        if self.fp16 and im.dtype != torch.float16:
            im = im.half()  # to FP16
        if self.nhwc:
            im = im.permute(0, 2, 3, 1)  # torch BCHW to numpy BHWC shape(1,320,192,3)

        if self.pt:  # PyTorch
            y = (
                self.model(im, augment=augment, visualize=visualize)
                if augment or visualize
                else self.model(im)
            )
        elif self.jit:  # TorchScript
            y = self.model(im)
        elif self.dnn:  # ONNX OpenCV DNN
            im = im.cpu().numpy()  # torch to numpy
            self.net.setInput(im)
            y = self.net.forward()
        elif self.onnx:  # ONNX Runtime
            im = im.cpu().numpy()  # torch to numpy
            y = self.session.run(
                self.output_names, {self.session.get_inputs()[0].name: im}
            )
        elif self.xml:  # OpenVINO
            im = im.cpu().numpy()  # FP32
            y = list(self.executable_network([im]).values())
        elif self.engine:  # TensorRT
            if self.dynamic and im.shape != self.bindings["images"].shape:
                i = self.model.get_binding_index("images")
                self.context.set_binding_shape(i, im.shape)  # reshape if dynamic
                self.bindings["images"] = self.bindings["images"]._replace(
                    shape=im.shape
                )
                for name in self.output_names:
                    i = self.model.get_binding_index(name)
                    self.bindings[name].data.resize_(
                        tuple(self.context.get_binding_shape(i))
                    )
            s = self.bindings["images"].shape
            assert (
                im.shape == s
            ), f"input size {im.shape} {'>' if self.dynamic else 'not equal to'} max model size {s}"
            self.binding_addrs["images"] = int(im.data_ptr())
            self.context.execute_v2(list(self.binding_addrs.values()))
            y = [self.bindings[x].data for x in sorted(self.output_names)]
        elif self.coreml:  # CoreML
            im = im.cpu().numpy()
            im = Image.fromarray((im[0] * 255).astype("uint8"))
            # im = im.resize((192, 320), Image.ANTIALIAS)
            y = self.model.predict({"image": im})  # coordinates are xywh normalized
            if "confidence" in y:
                box = xywh2xyxy(y["coordinates"] * [[w, h, w, h]])  # xyxy pixels
                conf, cls = y["confidence"].max(1), y["confidence"].argmax(1).astype(
                    np.float
                )
                y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
            else:
                y = list(
                    reversed(y.values())
                )  # reversed for segmentation models (pred, proto)
        elif self.paddle:  # PaddlePaddle
            im = im.cpu().numpy().astype(np.float32)
            self.input_handle.copy_from_cpu(im)
            self.predictor.run()
            y = [
                self.predictor.get_output_handle(x).copy_to_cpu()
                for x in self.output_names
            ]
        elif self.triton:  # NVIDIA Triton Inference Server
            y = self.model(im)
        else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
            im = im.cpu().numpy()
            if self.saved_model:  # SavedModel
                y = self.model(im, training=False) if self.keras else self.model(im)
            elif self.pb:  # GraphDef
                y = self.frozen_func(x=self.tf.constant(im))
            else:  # Lite or Edge TPU
                input = self.input_details[0]
                int8 = input["dtype"] == np.uint8  # is TFLite quantized uint8 model
                if int8:
                    scale, zero_point = input["quantization"]
                    im = (im / scale + zero_point).astype(np.uint8)  # de-scale
                self.interpreter.set_tensor(input["index"], im)
                self.interpreter.invoke()
                y = []
                for output in self.output_details:
                    x = self.interpreter.get_tensor(output["index"])
                    if int8:
                        scale, zero_point = output["quantization"]
                        x = (x.astype(np.float32) - zero_point) * scale  # re-scale
                    y.append(x)
            y = [x if isinstance(x, np.ndarray) else x.numpy() for x in y]
            y[0][..., :4] *= [w, h, w, h]  # xywh normalized to pixels

        if isinstance(y, (list, tuple)):
            return (
                self.from_numpy(y[0])
                if len(y) == 1
                else [self.from_numpy(x) for x in y]
            )
        else:
            return self.from_numpy(y)

    def from_numpy(self, x):
        return torch.from_numpy(x).to(self.device) if isinstance(x, np.ndarray) else x

    def warmup(self, imgsz=(1, 3, 640, 640), half=False):
        # Warmup model by running inference once
        warmup_types = (
            self.pt,
            self.jit,
            self.onnx,
            self.engine,
            self.saved_model,
            self.pb,
            self.triton,
        )
        if any(warmup_types) and (self.device.type != "cpu" or self.triton):
            im = torch.empty(
                *imgsz,
                dtype=torch.half if self.fp16 else torch.float,
                device=self.device,
            )  # input
            for _ in range(2 if self.jit else 1):  #
                self.forward(im)  # warmup

    @staticmethod
    def _model_type(p="path/to/model.pt"):
        # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
        # types = [pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, paddle]
        # from export import export_formats
        # from utils.downloads import is_url
        # from ta_pet_id.yolo.utils.downloads import is_url

        sf = list(export_formats().Suffix)  # export suffixes
        if not is_url(p, check=False):
            check_suffix(p, sf)  # checks
        url = urlparse(p)  # if url may be Triton inference server
        types = [s in Path(p).name for s in sf]
        types[8] &= not types[9]  # tflite &= not edgetpu
        triton = not any(types) and all(
            [any(s in url.scheme for s in ["http", "grpc"]), url.netloc]
        )
        return types + [triton]

    @staticmethod
    def _load_metadata(f=Path("path/to/meta.yaml")):
        # Load metadata from meta.yaml if it exists
        if f.exists():
            d = yaml_load(f)
            return d["stride"], d["names"]  # assign stride, names
        return None, None


################################################################
# class DetectMultiBackend(nn.Module):
#     # YOLOv5 MultiBackend class for python inference on various backends
#     def __init__(self, weights="yolov5s.pt", device=None, dnn=False):
#         from ta_pet_id.yolo.models.experimental import attempt_load

#         super().__init__()
#         w = str(weights[0] if isinstance(weights, list) else weights)
#         suffix = Path(w).suffix.lower()
#         suffixes = [
#             ".pt",
#             ".torchscript",
#             ".onnx",
#             ".engine",
#             ".tflite",
#             ".pb",
#             "",
#             ".mlmodel",
#         ]
#         check_suffix(w, suffixes)  # check weights have acceptable suffix
#         pt, jit, onnx, engine, tflite, pb, saved_model, coreml = (
#             suffix == x for x in suffixes
#         )  # backend booleans
#         stride, names = 64, [f"class{i}" for i in range(1000)]  # assign defaults

#         if pt:  # PyTorch
#             model = attempt_load(weights, map_location=device)
#             stride = int(model.stride.max())  # model stride
#             names = (
#                 model.module.names if hasattr(model, "module") else model.names
#             )  # get class names
#             self.model = model  # explicitly assign for to(), cpu(), cuda(), half()


#         else:  # TensorFlow model (TFLite, pb, saved_model)
#             print("some other extension")

#         self.__dict__.update(locals())  # assign all variables to self

#     def forward(self, im, augment=False, visualize=False, val=False):
#         # YOLOv5 MultiBackend inference
#         if self.pt:  # PyTorch
#             y = (
#                 self.model(im)
#                 if self.jit
#                 else self.model(im, augment=augment, visualize=visualize)
#             )
#             return y if val else y[0]


#         else:  # TensorFlow model (TFLite, pb, saved_model)
#             print("some other extension")
#         y = torch.tensor(y) if isinstance(y, np.ndarray) else y
#         return (y, []) if val else y

#     def warmup(self, imgsz=(1, 3, 640, 640), half=False):
#         # Warmup model by running inference once
#         if self.pt or self.engine or self.onnx:  # warmup types
#             if (
#                 isinstance(self.device, torch.device) and self.device.type != "cpu"
#             ):  # only warmup GPU models
#                 im = (
#                     torch.zeros(*imgsz)
#                     .to(self.device)
#                     .type(torch.half if half else torch.float)
#                 )  # input image
#                 self.forward(im)  # warmup


# class AutoShape(nn.Module):
#     # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
#     conf = 0.25  # NMS confidence threshold
#     iou = 0.45  # NMS IoU threshold
#     agnostic = False  # NMS class-agnostic
#     multi_label = False  # NMS multiple labels per box
#     classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
#     max_det = 1000  # maximum number of detections per image
#
#     def __init__(self, model):
#         super().__init__()
#         print('Adding AutoShape... ')
#         copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
#         self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
#         self.pt = not self.dmb or model.pt  # PyTorch model
#         self.model = model.eval()
#
#     def _apply(self, fn):
#         # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
#         self = super()._apply(fn)
#         if self.pt:
#             m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
#             m.stride = fn(m.stride)
#             m.grid = list(map(fn, m.grid))
#             if isinstance(m.anchor_grid, list):
#                 m.anchor_grid = list(map(fn, m.anchor_grid))
#         return self
#
#     @torch.no_grad()
#     def forward(self, imgs, size=640, augment=False, profile=False):
#         t = [time_sync()]
#         p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
#         if isinstance(imgs, torch.Tensor):  # torch
#             with amp.autocast(enabled=p.device.type != 'cpu'):
#                 return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference
#
#         # Pre-process
#         n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
#         shape0, shape1, files = [], [], []  # image and inference shapes, filenames
#         for i, im in enumerate(imgs):
#             f = f'image{i}'  # filename
#             if isinstance(im, (str, Path)):  # filename or uri
#                 im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
#                 im = np.asarray(exif_transpose(im))
#             elif isinstance(im, Image.Image):  # PIL Image
#                 im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
#             files.append(Path(f).with_suffix('.jpg').name)
#             if im.shape[0] < 5:  # image in CHW
#                 im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
#             im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
#             s = im.shape[:2]  # HWC
#             shape0.append(s)  # image shape
#             g = (size / max(s))  # gain
#             shape1.append([y * g for y in s])
#             imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
#         shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
#         x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]  # pad
#         x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
#         x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
#         x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
#         t.append(time_sync())
#
#         with amp.autocast(enabled=p.device.type != 'cpu'):
#             # Inference
#             y = self.model(x, augment, profile)  # forward
#             t.append(time_sync())
#
#             # Post-process
#             y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
#                                     agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
#             for i in range(n):
#                 scale_coords(shape1, y[i][:, :4], shape0[i])
#
#             t.append(time_sync())
#             return Detections(imgs, y, files, t, self.names, x.shape)


# class Detections:
#     # YOLOv5 detections class for inference results
#     def __init__(self, imgs, pred, files, times=None, names=None, shape=None):
#         super().__init__()
#         d = pred[0].device  # device
#         gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
#         self.imgs = imgs  # list of images as numpy arrays
#         self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
#         self.names = names  # class names
#         self.files = files  # image filenames
#         self.xyxy = pred  # xyxy pixels
#         self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
#         self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
#         self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
#         self.n = len(self.pred)  # number of images (batch size)
#         self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
#         self.s = shape  # inference BCHW shape
#
#     def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
#         crops = []
#         for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
#             s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
#             if pred.shape[0]:
#                 for c in pred[:, -1].unique():
#                     n = (pred[:, -1] == c).sum()  # detections per class
#                     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
#                 if show or save or render or crop:
#                     annotator = Annotator(im, example=str(self.names))
#                     for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
#                         label = f'{self.names[int(cls)]} {conf:.2f}'
#                         if crop:
#                             file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
#                             crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
#                                           'im': save_one_box(box, im, file=file, save=save)})
#                         else:  # all others
#                             annotator.box_label(box, label, color=colors(cls))
#                     im = annotator.im
#             else:
#                 s += '(no detections)'
#
#             im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
#             if pprint:
#                 LOGGER.info(s.rstrip(', '))
#             if show:
#                 im.show(self.files[i])  # show
#             if save:
#                 f = self.files[i]
#                 im.save(save_dir / f)  # save
#                 if i == self.n - 1:
#                     LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
#             if render:
#                 self.imgs[i] = np.asarray(im)
#         if crop:
#             if save:
#                 LOGGER.info(f'Saved results to {save_dir}\n')
#             return crops
#
#     def print(self):
#         self.display(pprint=True)  # print results
#         LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
#                     self.t)
#
#     def show(self):
#         self.display(show=True)  # show results
#
#     def save(self, save_dir='runs/detect/exp'):
#         save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
#         self.display(save=True, save_dir=save_dir)  # save results
#
#     def crop(self, save=True, save_dir='runs/detect/exp'):
#         save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
#         return self.display(crop=True, save=save, save_dir=save_dir)  # crop results
#
#     def render(self):
#         self.display(render=True)  # render results
#         return self.imgs
#
#     def pandas(self):
#         # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
#         new = copy(self)  # return copy
#         ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
#         cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
#         for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
#             a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
#             setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
#         return new
#
#     def tolist(self):
#         # return a list of Detections objects, i.e. 'for result in results.tolist():'
#         x = [Detections([self.imgs[i]], [self.pred[i]], self.names, self.s) for i in range(self.n)]
#         for d in x:
#             for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
#                 setattr(d, k, getattr(d, k)[0])  # pop out of list
#         return x
#
#     def __len__(self):
#         return self.n


# class Classify(nn.Module):
#     # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
#     def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
#         super().__init__()
#         self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
#         self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
#         self.flat = nn.Flatten()
#
#     def forward(self, x):
#         z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
#         return self.flat(self.conv(z))  # flatten to x(b,c2)
