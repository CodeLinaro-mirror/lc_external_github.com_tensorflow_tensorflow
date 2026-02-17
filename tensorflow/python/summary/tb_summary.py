# Copyright 2023 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Re-exports the APIs of TF2 summary that live in TensorBoard."""

from tensorflow.python.util.tf_export import tf_export

_TENSORBOARD_NOT_INSTALLED_ERROR = (
    "TensorBoard is not installed, missing implementation for"
)


class TBNotInstalledError(Exception):
  def __init__(self, summary_api):
    self.error_message = f"{_TENSORBOARD_NOT_INSTALLED_ERROR} {summary_api}"
    super().__init__(self.error_message)

try:
  from tensorboard.summary.v2 import audio as audio_v2  # pylint: disable=g-import-not-at-top
  from tensorboard.summary.v2 import histogram as histogram_v2  # pylint: disable=g-import-not-at-top
  from tensorboard.summary.v2 import image as image_v2  # pylint: disable=g-import-not-at-top
  from tensorboard.summary.v2 import scalar as scalar_v2  # pylint: disable=g-import-not-at-top
  from tensorboard.summary.v2 import text as text_v2  # pylint: disable=g-import-not-at-top
  _TENSORBOARD_AVAILABLE = True
except ImportError:
  def _tb_not_installed(*args, **kwargs):
    raise TBNotInstalledError("tf.summary")
  audio_v2 = _tb_not_installed
  histogram_v2 = _tb_not_installed
  image_v2 = _tb_not_installed
  scalar_v2 = _tb_not_installed
  text_v2 = _tb_not_installed
  _TENSORBOARD_AVAILABLE = False

audio = tf_export("summary.audio", v1=[])(audio_v2)
histogram = tf_export("summary.histogram", v1=[])(histogram_v2)
image = tf_export("summary.image", v1=[])(image_v2)
scalar = tf_export("summary.scalar", v1=[])(scalar_v2)
text = tf_export("summary.text", v1=[])(text_v2)
