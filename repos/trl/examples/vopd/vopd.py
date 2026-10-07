# Copyright 2020-2026 The HuggingFace Team. All rights reserved.
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

"""Run vOPD with the standard distillation CLI, including its LoRA and dataset options."""

from trl.scripts.distillation import main, make_parser


if __name__ == "__main__":
    parser = make_parser()
    script_args, training_args, model_args, dataset_args = parser.parse_args_and_config()
    if training_args.loss_type != "vopd":
        raise ValueError("Pass --loss_type vopd to this example.")
    main(script_args, training_args, model_args, dataset_args)
