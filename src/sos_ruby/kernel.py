#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

from sos.utils import short_repr, env
import json

Ruby_init_statement = '''


'''

#
#  support for %get
#
#  Converting a Python object to a JSON format to be loaded by Ruby
#
def _JS_repr(obj):
    try:
        # for JSON serlializable type, we simply dump it
        return json.dumps(obj)
    except Exception:
        import numpy
        import pandas
        if isinstance(obj, (numpy.intc, numpy.intp, numpy.int8, numpy.int16, numpy.int32, numpy.int64,\
                numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float16, numpy.float32, \
                numpy.float64, numpy.matrixlib.defmatrix.matrix, numpy.ndarray)):
            return json.dumps(obj.tolist())
        elif isinstance(obj, pandas.DataFrame):
            return obj.to_json(orient='index')
        elif isinstance(obj, set):
            return json.dumps(list(obj))
        else:
            return 'Unsupported seralizable data {} with type {}'.format(short_repr(obj), obj.__class__.__name__)


class sos_Ruby:
    supported_kernels = {'Ruby': ['ruby']}
    background_color = '#EA5745'
    options = {}

    def __init__(self, sos_kernel, kernel_name='ruby'):
        self.sos_kernel = sos_kernel
        self.kernel_name = kernel_name
        self.init_statements = Ruby_init_statement

    def get_vars(self, names):
        for name in names:
            self.sos_kernel.run_cell('{} = {}'.format(name, _JS_repr(env.sos_dict[name])), True, False)

    def put_vars(self, items, to_kernel=None):
        # first let us get all variables with names starting with sos
        response = self.sos_kernel.get_response('__get_sos_vars()', ('execute_result'))[0][1]
        expr = response['data']['text/plain']
        items += eval(expr)

        if not items:
            return {}

        py_repr = 'JSON.stringify({{ {} }})'.format(','.join('"{0}":{0}'.format(x) for x in items))
        response = self.sos_kernel.get_response(py_repr, ('execute_result'))[0][1]
        expr = response['data']['text/plain']
        try:
            return json.loads(eval(expr))
        except Exception as e:
            self.sos_kernel.warn('Failed to convert {} to Python object: {}'.format(expr, e))
            return {}
