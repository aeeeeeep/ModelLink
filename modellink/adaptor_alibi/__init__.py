# coding=utf-8
#
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

from .model_megatron_adaptor import apply_model_patch
from .arguments_adaptor import apply_arguments_patch


def apply_alibi_patch():
    apply_arguments_patch()
    apply_model_patch()
