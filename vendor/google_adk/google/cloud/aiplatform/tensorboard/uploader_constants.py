"""Constants shared between TensorBoard command line uploader and SDK uploader"""

import dataclasses

from tensorboard.plugins.distribution import distributions_plugin
from tensorboard.plugins.graph import graphs_plugin
from tensorboard.plugins.histogram import histograms_plugin
from tensorboard.plugins.hparams import hparams_plugin
from tensorboard.plugins.image import images_plugin
from tensorboard.plugins.scalar import scalars_plugin
from tensorboard.plugins.text import text_plugin

PROFILE_PLUGIN_NAME = "profile"
ALLOWED_PLUGINS = frozenset(
    [
        scalars_plugin.ScalarsPlugin.plugin_name,
        histograms_plugin.HistogramsPlugin.plugin_name,
        distributions_plugin.DistributionsPlugin.plugin_name,
        text_plugin.TextPlugin.plugin_name,
        hparams_plugin.HParamsPlugin.plugin_name,
        images_plugin.ImagesPlugin.plugin_name,
        graphs_plugin.GraphsPlugin.plugin_name,
        PROFILE_PLUGIN_NAME,
    ]
)

# Minimum length of a logdir polling cycle in seconds. Shorter cycles will
# sleep to avoid spinning over the logdir, which isn't great for disks and can
# be expensive for network file systems.
MIN_LOGDIR_POLL_INTERVAL_SECS = 1

# Maximum length of a base-128 varint as used to encode a 64-bit value
# (without the "msb of last byte is bit 63" optimization, to be
# compatible with protobuf and golang varints).
MAX_VARINT64_LENGTH_BYTES = 10

# Default minimum interval between initiating WriteTensorbordRunData RPCs in
# milliseconds.
DEFAULT_MIN_SCALAR_REQUEST_INTERVAL = 30

# Default maximum WriteTensorbordRunData request size in bytes.
DEFAULT_MAX_SCALAR_REQUEST_SIZE = 64 * (2**10)  # 64KiB

# Default minimum interval between initiating WriteTensorbordRunData RPCs in
# milliseconds.
DEFAULT_MIN_TENSOR_REQUEST_INTERVAL = 30

# Default minimum interval between initiating WriteTensorbordRunData RPCs in
# milliseconds.
DEFAULT_MIN_BLOB_REQUEST_INTERVAL = 30

# Default maximum WriteTensorbordRunData request size in bytes.
DEFAULT_MAX_TENSOR_REQUEST_SIZE = 64 * (2**10)  # 64KiB

DEFAULT_MAX_BLOB_REQUEST_SIZE = 64 * (2**10)  # 64KiB

# Default maximum tensor point size in bytes.
DEFAULT_MAX_TENSOR_POINT_SIZE = 16 * (2**10)  # 16KiB

DEFAULT_MAX_BLOB_SIZE = 10 * (2**30)  # 10GiB


@dataclasses.dataclass
class UploadLimits:
    max_scalar_request_size: int = DEFAULT_MAX_SCALAR_REQUEST_SIZE
    max_tensor_request_size: int = DEFAULT_MAX_TENSOR_REQUEST_SIZE
    max_blob_request_size: int = DEFAULT_MAX_BLOB_REQUEST_SIZE

    min_scalar_request_interval: int = DEFAULT_MIN_SCALAR_REQUEST_INTERVAL
    min_tensor_request_interval: int = DEFAULT_MIN_TENSOR_REQUEST_INTERVAL
    min_blob_request_interval: int = DEFAULT_MIN_BLOB_REQUEST_INTERVAL

    max_blob_size: int = DEFAULT_MAX_BLOB_SIZE
    max_tensor_point_size: int = DEFAULT_MAX_TENSOR_POINT_SIZE
