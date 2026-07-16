# noise_argparser.py
# 模块职责：解析噪声层配置字符串，并将其转换为对应的噪声层模块实例。
# 支持的噪声配置格式（trainer.py 中用 '+' 拆分组合，Noiser 每次 forward 随机选择一个激活噪声层）：
#   - crop((hmin,hmax),(wmin,wmax))      ：从图像中心裁剪指定高度/宽度比例范围
#   - cropout((hmin,hmax),(wmin,wmax))   ：在图像外围填充/裁剪指定比例范围
#   - dropout(keep_min,keep_max)         ：按保留比例范围随机丢弃像素
#   - resize(min_ratio,max_ratio)        ：按比例范围缩放图像
#   - jpeg                               ：JPEG 压缩噪声
#   - quant                              ：量化噪声
# 本模块还提供 NoiseArgParser（argparse.Action 子类），用于在 train.py 中作为 --noise 参数的 action，
# 将原始字符串配置直接存入 namespace，便于序列化与后续 Noiser 解析。
import argparse  # 命令行参数解析，NoiseArgParser 继承自 argparse.Action
import re  # 正则表达式，用于解析噪声配置字符串中的数值
# 以下为各噪声层模块，每个模块都是一个可微或随机的信道噪声层
from noise_layers.cropout import Cropout  # cropout 噪声：外围裁剪/填充
from noise_layers.crop import Crop  # crop 噪声：中心裁剪
from noise_layers.identity import Identity  # identity 噪声：恒等映射（无噪声）
from noise_layers.dropout import Dropout  # dropout 噪声：随机丢弃像素
from noise_layers.resize import Resize  # resize 噪声：随机缩放
from noise_layers.quantization import Quantization  # quant 噪声：量化噪声
from noise_layers.jpeg_compression import JpegCompression  # jpeg 噪声：JPEG 压缩


def parse_pair(match_groups):
    # 从正则匹配的两个分组中解析高度范围与宽度范围。
    # 参数：match_groups - 正则匹配结果元组，格式为 (高度范围字符串, 宽度范围字符串)
    # 返回：((hmin, hmax), (wmin, wmax)) 两个范围元组
    # 解析高度范围：按逗号拆分为最小值与最大值
    heights = match_groups[0].split(',')
    hmin = float(heights[0])
    hmax = float(heights[1])
    # 解析宽度范围：按逗号拆分为最小值与最大值
    widths = match_groups[1].split(',')
    wmin = float(widths[0])
    wmax = float(widths[1])
    return (hmin, hmax), (wmin, wmax)


def parse_crop(crop_command):
    # 解析 crop((hmin,hmax),(wmin,wmax)) 格式的配置字符串，返回 Crop 噪声层实例。
    # 参数：crop_command - 形如 'crop((0.1,0.2),(0.1,0.2))' 的字符串
    # 返回：Crop 噪声层实例
    # 抛出：ValueError - 格式不匹配时
    # 正则匹配两个数值范围分组（每组为 逗号分隔的两个数值）
    matches = re.match(r'crop\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', crop_command)
    if matches is None:
        # 格式错误时抛出异常并给出期望格式提示
        raise ValueError(f'Invalid crop command format: {crop_command}. Expected format: crop((hmin,hmax),(wmin,wmax))')
    # 解析高度与宽度范围
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Crop((hmin, hmax), (wmin, wmax))

def parse_cropout(cropout_command):
    # 解析 cropout((hmin,hmax),(wmin,wmax)) 格式的配置字符串，返回 Cropout 噪声层实例。
    # 参数：cropout_command - 形如 'cropout((0.55,0.6),(0.55,0.6))' 的字符串
    # 返回：Cropout 噪声层实例
    # 抛出：ValueError - 格式不匹配时
    # 正则匹配两个数值范围分组
    matches = re.match(r'cropout\(\((\d+\.*\d*,\d+\.*\d*)\),\((\d+\.*\d*,\d+\.*\d*)\)\)', cropout_command)
    if matches is None:
        # 格式错误时抛出异常
        raise ValueError(f'Invalid cropout command format: {cropout_command}. Expected format: cropout((hmin,hmax),(wmin,wmax))')
    # 解析高度与宽度范围
    (hmin, hmax), (wmin, wmax) = parse_pair(matches.groups())
    return Cropout((hmin, hmax), (wmin, wmax))


def parse_dropout(dropout_command):
    # 解析 dropout(keep_min,keep_max) 格式的配置字符串，返回 Dropout 噪声层实例。
    # 参数：dropout_command - 形如 'dropout(0.3,0.5)' 的字符串
    # 返回：Dropout 噪声层实例
    # 抛出：ValueError - 格式不匹配时
    # 正则匹配单个保留比例范围分组
    matches = re.match(r'dropout\((\d+\.*\d*,\d+\.*\d*)\)', dropout_command)
    if matches is None:
        # 格式错误时抛出异常
        raise ValueError(f'Invalid dropout command format: {dropout_command}. Expected format: dropout(keep_min,keep_max)')
    # 解析保留比例范围：按逗号拆分为最小与最大保留比例
    ratios = matches.groups()[0].split(',')
    keep_min = float(ratios[0])
    keep_max = float(ratios[1])
    return Dropout((keep_min, keep_max))

def parse_resize(resize_command):
    # 解析 resize(min_ratio,max_ratio) 格式的配置字符串，返回 Resize 噪声层实例。
    # 参数：resize_command - 形如 'resize(0.7,1.2)' 的字符串
    # 返回：Resize 噪声层实例
    # 抛出：ValueError - 格式不匹配时
    # 正则匹配单个缩放比例范围分组
    matches = re.match(r'resize\((\d+\.*\d*,\d+\.*\d*)\)', resize_command)
    if matches is None:
        # 格式错误时抛出异常
        raise ValueError(f'Invalid resize command format: {resize_command}. Expected format: resize(min_ratio,max_ratio)')
    # 解析缩放比例范围：按逗号拆分为最小与最大比例
    ratios = matches.groups()[0].split(',')
    min_ratio = float(ratios[0])
    max_ratio = float(ratios[1])
    return Resize((min_ratio, max_ratio))


class NoiseArgParser(argparse.Action):
    # 自定义 argparse Action，用于解析 --noise 参数。
    # 作用：将原始噪声配置字符串（而非已解析的噪声层对象）存入 namespace，
    # 这样配置字符串可被序列化到 config.json，并在 Trainer 中再由 Noiser 解析为噪声层组合。
    def __init__(self,
                 option_strings,
                 dest,
                 nargs=None,
                 const=None,
                 default=None,
                 type=None,
                 choices=None,
                 required=False,
                 help=None,
                 metavar=None):
        # 调用父类 argparse.Action 的初始化方法，转发全部参数
        argparse.Action.__init__(self,
                                 option_strings=option_strings,
                                 dest=dest,
                                 nargs=nargs,
                                 const=const,
                                 default=default,
                                 type=type,
                                 choices=choices,
                                 required=required,
                                 help=help,
                                 metavar=metavar,
                                 )

    @staticmethod
    def parse_cropout_args(cropout_args):
        # 占位方法：预留用于解析 cropout 参数（当前未实现，pass 保持空实现）
        pass

    @staticmethod
    def parse_dropout_args(dropout_args):
        # 占位方法：预留用于解析 dropout 参数（当前未实现，pass 保持空实现）
        pass

    def __call__(self, parser, namespace, values,
                 option_string=None):
        # argparse Action 的回调：在解析到 --noise 参数时被调用。
        # 参数：
        #   parser - 解析器实例
        #   namespace - 命名空间，用于存储解析结果
        #   values - 参数值列表（nargs='*' 时为列表，取第一个元素为配置字符串）
        #   option_string - 选项字符串（如 '--noise'）
        # 存储原始字符串配置而非已解析的噪声层对象，
        # 这样便于序列化到 config.json，并避免后续 join() 错误
        noise_config = values[0]
        # 将噪声配置字符串写入 namespace 的目标属性
        setattr(namespace, self.dest, noise_config)
