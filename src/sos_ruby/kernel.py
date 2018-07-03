#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

from collections import Sequence
from sos.utils import short_repr, env
import json

Ruby_init_statement = '''
def __Ruby_py_repr(obj)
  if obj.instance_of? Integer
    return obj.inspect
  elsif obj.instance_of? String
    return obj
  elsif obj.instance_of? TrueClass
    return "True"
  elsif obj.instance_of? FalseClass
    return "False"
  elsif obj.instance_of? Float
    return obj.inspect
  elsif obj.nil?
    return "None"
  elsif obj.instance_of? Range
    return "range(" + obj.min().inspect + "," + (obj.max()+1).inspect + ")"
  elsif obj.instance_of? Array
    return obj.map { |indivial_var| __Ruby_py_repr(indivial_var) }
  elsif obj.instance_of? Daru::DataFrame
    #return "import pandas\n" + "pandas.DataFrame(" + "{" + obj.vectors.to_a.map{|x| "\"" + x.to_s + "\":" + obj[x].to_a.map{|y|  __Ruby_py_repr(y)}.to_s}.join(",") + "})"
    return "pandas.DataFrame(" + "{" + obj.vectors.to_a.map{|x| "\"" + x.to_s + "\":" + obj[x].to_a.map{|y|  __Ruby_py_repr(y)}.to_s}.join(",") + "})"
  elsif obj.instance_of? NMatrix
    return "numpy.matrix(" + obj.to_a.to_s + ")"
  end
end
'''

#
#  support for %get
#
#  Converting a Python object to a JSON format to be loaded by Ruby
#





class sos_Ruby:
    supported_kernels = {'Ruby': ['Ruby']}
    background_color = '#EA5745'
    options = {}

    def __init__(self, sos_kernel, kernel_name='Ruby'):
        self.sos_kernel = sos_kernel
        self.kernel_name = kernel_name
        self.init_statements = Ruby_init_statement

    def _Ruby_repr(self, obj):
        if isinstance(obj, bool):
            return 'true' if obj else 'false'
        elif isinstance(obj, (int, float)):
            return repr(obj)
        elif isinstance(obj, str):
            return '"""' + obj + '"""'
        elif isinstance(obj, complex):
            return 'Complex(' + str(obj.real) + ',' + str(obj.imag) + ')'
        elif isinstance(obj, range):
            return '(' + repr(min(obj)) + '...' + repr(max(obj)) + ')'
        elif isinstance(obj, Sequence):
            if len(obj) == 0:
                return '[]'
            else:
                return '[' + ','.join(self._Ruby_repr(x) for x in obj) + ']'
        elif obj is None:
            return 'nil'
        elif isinstance(obj, dict):
            return '{' + ','.join('"{}" => {}'.format(x, self.Ruby_repr(y)) for x,y in obj.items()) + '}'
        elif isinstance(obj, set):
            return 'Set[' + ','.join(self._Ruby_repr(x) for x in obj) + ']'
        '''
        else:
            if isinstance(obj, (numpy.intc, numpy.intp, numpy.int8, numpy.int16, numpy.int32, numpy.int64,\
                    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float16, numpy.float32)):
                return repr(obj)
            # need to specify Float64() as the return to Julia in order to avoid losing precision
            elif isinstance(obj, numpy.float64):
                return 'Float64(' + obj + ')'
            elif isinstance(obj, numpy.matrixlib.defmatrix.matrix):
                try:
                    import feather
                except ImportError:
                    raise UsageError('The feather-format module is required to pass numpy matrix as julia matrix(array)'
                        'See https://github.com/wesm/feather/tree/master/python for details.')
                feather_tmp_ = tempfile.NamedTemporaryFile(suffix='.feather', delete=False).name
                feather.write_dataframe(pandas.DataFrame(obj).copy(), feather_tmp_)
                return 'Array(Feather.read("' + feather_tmp_ + '", nullable=false))'
            elif isinstance(obj, numpy.ndarray):
                return '[' + ','.join(self._julia_repr(x) for x in obj) + ']'
            elif isinstance(obj, pandas.DataFrame):
                try:
                    import feather
                except ImportError:
                    raise UsageError('The feather-format module is required to pass pandas DataFrame as julia.DataFrames'
                        'See https://github.com/wesm/feather/tree/master/python for details.')
                feather_tmp_ = tempfile.NamedTemporaryFile(suffix='.feather', delete=False).name
                try:
                    data = obj.copy()
                    # Julia DataFrame does not have index
                    if not isinstance(data.index, pandas.RangeIndex):
                        self.sos_kernel.warn('Raw index is ignored because Julia DataFrame does not support raw index.') 
                    feather.write_dataframe(data, feather_tmp_)
                except Exception:
                    # if data cannot be written, we try to manipulate data
                    # frame to have consistent types and try again
                    for c in data.columns:
                        if not homogeneous_type(data[c]):
                            data[c] = [str(x) for x in data[c]]
                    feather.write_dataframe(data, feather_tmp_)
                    # use {!r} for path because the string might contain c:\ which needs to be
                    # double quoted.
                return 'Feather.read("' + feather_tmp_ + '", nullable=false)'
            elif isinstance(obj, pandas.Series):
                dat=list(obj.values)
                ind=list(obj.index.values)
                ans='NamedArray(' + '[' + ','.join(self._julia_repr(x) for x in dat) + ']' + ',([' + ','.join(self._julia_repr(y) for y in ind) + '],))'
                return ans.replace("'",'"')
            else:
                return repr('Unsupported datatype {}'.format(short_repr(obj)))
        '''




    def get_vars(self, names):
        for name in names:
            self.sos_kernel.run_cell('{} = {}'.format(name, _Ruby_repr(env.sos_dict[name])), True, False)

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
