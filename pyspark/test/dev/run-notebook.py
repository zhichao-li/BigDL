#!/usr/bin/env python


#
# Copyright 2016 The BigDL Authors.
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
#


import os
import subprocess
import tempfile

import nbformat

import re
def fail_methods(nb):
    fail_method_re = r"""class="nosefailedfunc">(.*)</span>"""
    for cell in nb.cells:
        if "outputs" in cell:
            for output in cell["outputs"]:
                if output.output_type == "execute_result":
                    text = str(output["data"])
                    captureMethods = re.search(fail_method_re, text).groups()  #__main__.test_arithmetic
                    for group in captureMethods:
                        yield group

def run_notebook(path):
    dirname, __ = os.path.split(path)
    os.chdir(dirname)
    
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as fout:
        args = ["jupyter", "nbconvert", "--to", "notebook", "--execute",
          "--ExecutePreprocessor.timeout=360",
          "--output", fout.name, path]
        subprocess.check_call(args)

        fout.seek(0)
        nb = nbformat.read(fout, nbformat.current_nbformat)
        return [m for m in fail_methods(nb)]

def test_notebook(path):
    errors = run_notebook(path)
    for method in fail_methods(nb):
        print("Fail method: %s" % method)

    if errors:
        raise Exception("Test fail for this notebook")

notebooks = ["/home/lisurprise/bin/god/BigDL/pyspark/dl/example/tutorial/simple_text_classification/text_classfication.ipynb"]
for notebook in notebooks:
    test_notebook(notebook)

