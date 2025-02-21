# Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

import os
import sys

__dir__ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(__dir__)
sys.path.append(os.path.abspath(os.path.join(__dir__, '..')))

os.environ["FLAGS_allocator_strategy"] = 'auto_growth'

import paddle

from ppocr.data import create_operators, transform
from ppocr.modeling.architectures import build_model
from ppocr.postprocess import build_post_process
from ppocr.utils.save_load import init_model
from ppocr.utils.utility import get_image_file_list
import tools.program as program


def main():
    global_config = config['Global']

    # build post process
    post_process_class = build_post_process(config['PostProcess'],
                                            global_config)

    # build model
    if hasattr(post_process_class, 'character'):
        config['Architecture']["Head"]['out_channels'] = len(
            getattr(post_process_class, 'character'))

    model = build_model(config['Architecture'])

    init_model(config, model, logger)

    # create data ops
    transforms = []
    for op in config['Eval']['dataset']['transforms']:
        op_name = list(op)[0]
        if 'Label' in op_name:
            continue
        elif op_name in ['RecResizeImg']:
            op[op_name]['infer_mode'] = True
        elif op_name == 'KeepKeys':
            if config['Architecture']['algorithm'] == "SRN":
                op[op_name]['keep_keys'] = [
                    'image', 'encoder_word_pos', 'gsrm_word_pos',
                    'gsrm_slf_attn_bias1', 'gsrm_slf_attn_bias2'
                ]
            else:
                op[op_name]['keep_keys'] = ['image']
        transforms.append(op)
    global_config['infer_mode'] = True
    ops = create_operators(transforms, global_config)

    save_res_path = config['Global'].get('save_res_path',
                                         "./output/rec/predicts_rec.txt")
    if not os.path.exists(os.path.dirname(save_res_path)):
        os.makedirs(os.path.dirname(save_res_path))

    model.eval()
    input_spec = paddle.static.InputSpec(shape=[None, 3, 32, 280], dtype='float32', name='image')
    file_path = "/tmp/rec_mv3_none_none_ctc_20210714"
    paddle.onnx.export(model, file_path, input_spec=[input_spec], opset_version=12)
    import onnx
    model2 = onnx.load(file_path + ".onnx")
    model2.graph.input[0].type.tensor_type.shape.dim[0].dim_param = '?'
    model2.graph.input[0].type.tensor_type.shape.dim[3].dim_param = '?'
    onnx.save(model2, file_path + "_dynamic.onnx")


    # with open(save_res_path, "w") as fout:
    #     for file in get_image_file_list(config['Global']['infer_img']):
    #         logger.info("infer_img: {}".format(file))
    #         with open(file, 'rb') as f:
    #             img = f.read()
    #             data = {'image': img}
    #         batch = transform(data, ops)
    #         if config['Architecture']['algorithm'] == "SRN":
    #             encoder_word_pos_list = np.expand_dims(batch[1], axis=0)
    #             gsrm_word_pos_list = np.expand_dims(batch[2], axis=0)
    #             gsrm_slf_attn_bias1_list = np.expand_dims(batch[3], axis=0)
    #             gsrm_slf_attn_bias2_list = np.expand_dims(batch[4], axis=0)
    #
    #             others = [
    #                 paddle.to_tensor(encoder_word_pos_list),
    #                 paddle.to_tensor(gsrm_word_pos_list),
    #                 paddle.to_tensor(gsrm_slf_attn_bias1_list),
    #                 paddle.to_tensor(gsrm_slf_attn_bias2_list)
    #             ]
    #
    #         images = np.expand_dims(batch[0], axis=0)
    #         images = paddle.to_tensor(images)
    #         if config['Architecture']['algorithm'] == "SRN":
    #             preds = model(images, others)
    #         else:
    #             preds = model(images)
    #         post_result = post_process_class(preds)
    #         for rec_result in post_result:
    #             logger.info('\t result: {}'.format(rec_result))
    #             if len(rec_result) >= 2:
    #                 fout.write(file + "\t" + rec_result[0] + "\t" + str(
    #                     rec_result[1]) + "\n")
    # logger.info("success!")


if __name__ == '__main__':
    config, device, logger, vdl_writer = program.preprocess()
    main()
