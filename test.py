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

import json
import re

import paddlehub as hub

image_path = ["./garbage/22/img_10962.jpg", ]
top_k = 1
module = hub.Module(name="garbage_classification")

filepath='/home/aistudio/The-Eye-Konws-the-Garbage/garbage_classification.json'

res = module.predict(paths=image_path, top_k=top_k)

for i, image in enumerate(image_path):

    print("The returned result of {}: {}".format(image, res[i]))
    print(res[i])
    category_id = res[i][0][0]
    score = res[i][1][0]
    time = res[i][2]
    print(category_id)
    print(score)
    print(time)
    f_obj=open(filepath)
    content=json.load(f_obj)[str(category_id)]
    print(content)

