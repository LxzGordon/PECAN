import {OpMapper} from '../types';

/**
 * @license
 * Copyright 2018 Google LLC. All Rights Reserved.
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 * =============================================================================
 */

export const json: OpMapper[] = [
  {
    'tfOpName': 'AvgPool',
    'category': 'convolution',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'strides', 'name': 'strides', 'type': 'number[]'},
      {'tfName': 'padding', 'name': 'pad', 'type': 'string'}, {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'notSupported': true
      },
      {'tfName': 'ksize', 'name': 'kernelSize', 'type': 'number[]'},
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'MaxPool',
    'category': 'convolution',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'strides', 'name': 'strides', 'type': 'number[]'},
      {'tfName': 'padding', 'name': 'pad', 'type': 'string'}, {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'notSupported': true
      },
      {'tfName': 'ksize', 'name': 'kernelSize', 'type': 'number[]'},
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}
    ]
  },
  {
    'tfOpName': 'Conv1D',
    'category': 'convolution',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'filter', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'stride', 'name': 'stride', 'type': 'number'},
      {'tfName': 'padding', 'name': 'pad', 'type': 'string'}, {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'defaultValue': 'NWC'
      },
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true}, {
        'tfName': 'dilation',
        'name': 'dilation',
        'type': 'number',
        'defaultValue': 1
      }
    ]
  },
  {
    'tfOpName': 'Conv2D',
    'category': 'convolution',
    'inputs': [
      {'start': 0, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'filter', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'T', 'name': 'dtype', 'type': 'dtype', 'notSupported': true},
      {'tfName': 'strides', 'name': 'strides', 'type': 'number[]'},
      {'tfName': 'padding', 'name': 'pad', 'type': 'string'},
      {'tfName': 'useCudnnOnGpu', 'name': 'useCudnnOnGpu', 'type': 'bool'}, {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'defaultValue': 'NHWC'
      },
      {'tfName': 'dilations', 'name': 'dilations', 'type': 'number[]'}
    ]
  },
  {
    'tfOpName': 'Conv2DBackpropInput',
    'category': 'convolution',
    'inputs': [
      {'start': 2, 'name': 'x', 'type': 'tensor'},
      {'start': 1, 'name': 'filter', 'type': 'tensor'},
      {'start': 0, 'name': 'outputShape', 'type': 'number[]'},
    ],
    'attrs': [
      {'tfName': 'strides', 'name': 'strides', 'type': 'number[]'},
      {'tfName': 'padding', 'name': 'pad', 'type': 'string'}, {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'notSupported': true
      }
    ]
  },
  {
    'tfOpName': 'DepthwiseConv2d',
    'category': 'convolution',
    'inputs': [
      {'start': 0, 'name': 'input', 'type': 'tensor'},
      {'start': 1, 'name': 'filter', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'strides', 'name': 'strides', 'type': 'number[]'},
      {'tfName': 'padding', 'name': 'pad', 'type': 'string'}, {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'defaultValue': 'NHWC'
      },
      {'tfName': 'dilations', 'name': 'dilations', 'type': 'number[]'}
    ]
  },
  {
    'tfOpName': 'DepthwiseConv2dNative',
    'category': 'convolution',
    'inputs': [
      {'start': 0, 'name': 'input', 'type': 'tensor'},
      {'start': 1, 'name': 'filter', 'type': 'tensor'},
    ],
    'attrs': [
      {'tfName': 'strides', 'name': 'strides', 'type': 'number[]'},
      {'tfName': 'padding', 'name': 'pad', 'type': 'string'}, {
        'tfName': 'data_format',
        'name': 'dataFormat',
        'type': 'string',
        'defaultValue': 'NHWC'
      },
      {'tfName': 'dilations', 'name': 'dilations', 'type': 'number[]'}
    ]
  }
];
