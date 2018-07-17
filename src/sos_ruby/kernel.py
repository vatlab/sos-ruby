#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

from collections import Sequence
from sos.utils import short_repr, env, log_to_file
import numpy
import pandas
import json

Ruby_init_statement = '''
require 'daru'
require 'nmatrix'
def __Ruby_py_repr(obj)
  if obj.instance_of? Integer
    return obj.inspect
  elsif obj.instance_of? String
    return obj.inspect
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
    return '[' + (obj.map { |indivial_var| __Ruby_py_repr(indivial_var) } ).join(",") + ']'
  elsif obj.instance_of? Daru::DataFrame
    return "pandas.DataFrame(" + "{" + (obj.vectors.to_a.map { |x| "\"" + x.to_s + "\":" + (obj[x].to_a.map { |y|  eval(__Ruby_py_repr(y)) }).to_s } ).join(",") + "}," + "index=" + obj.index.to_a.to_s + ")"
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
    supported_kernels = {'Ruby': ['ruby']}
    background_color = '#e8c2be'
    options = {}
    cd_command = 'Dir.chdir {dir!r}'

    def __init__(self, sos_kernel, kernel_name='ruby'):
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
            return '{' + ','.join('"{}" => {}'.format(x, self._Ruby_repr(y)) for x, y in obj.items()) + '}'
        elif isinstance(obj, set):
            return 'Set[' + ','.join(self._Ruby_repr(x) for x in obj) + ']'
        else:
            if isinstance(obj, (numpy.intc, numpy.intp, numpy.int8, numpy.int16, numpy.int32, numpy.int64,\
                    numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float16, numpy.float32, numpy.float64)):
                return repr(obj)
            elif isinstance(obj, numpy.matrixlib.defmatrix.matrix):
                return 'N' + repr(obj.tolist())
            elif isinstance(obj, numpy.ndarray):
                return repr(obj.tolist())
            elif isinstance(obj, pandas.DataFrame):
                _beginning_result_string_dataframe_to_ruby = "Daru::DataFrame.new({"
                _context_string_dataframe_to_ruby = str(['"'
                                                        + str(x).replace("'", '"')
                                                        + '"'
                                                        + "=>"
                                                        + str(list(map(lambda y: eval(self._Ruby_repr(y)),
                                                                       obj[x].tolist()))).replace("'", '"')
                                                        for x in obj.keys().tolist()])[2:-2].replace("\', \'", ", ") + "},"
                _indexing_result_string_dataframe_to_ruby = "index:" + str(obj.index.values.tolist()).replace("'", '"') + ")"
                _result_string_dataframe_to_ruby = _beginning_result_string_dataframe_to_ruby + _context_string_dataframe_to_ruby + _indexing_result_string_dataframe_to_ruby
                return _result_string_dataframe_to_ruby
            elif isinstance(obj, pandas.Series):
                dat=list(obj.values)
                ind=list(obj.index.values)
                ans="{" + ",".join([repr(x) + "=>" + repr(y) for x, y in zip(ind, dat)]) + "}"
                return ans
            else:
                return repr('Unsupported datatype {}'.format(short_repr(obj)))

    def get_vars(self, names):
        for name in names:
            newname = name
            ruby_repr = self._Ruby_repr(env.sos_dict[name])
            self.sos_kernel.run_cell('{} = {}'.format(newname, ruby_repr), True, False,
                                     on_error='Failed to put variable {} to Ruby'.format(name))

    def put_vars(self, items, to_kernel=None):
        # first let us get all variables with names starting with sos
        try:
            response = self.sos_kernel.get_response('print local_variables', ('stream',), name=('stdout',))[0][1]
            all_vars = response['text']
            items += [x for x in all_vars[1:-1].split(", ") if x.startswith(":sos")]
        except:
            # if there is no variable with name sos, the command will not produce any output
            pass
        res = {}
        for item in items:
            py_repr = 'printf(__Ruby_py_repr({}))'.format(item)
            response = self.sos_kernel.get_response(py_repr, ('stream',), name=('stdout',))[0][1]
            expr = response['text']
            self.sos_kernel.warn(repr(expr))

            try:
                # evaluate as raw string to correctly handle \\ etc
                res[item] = eval(expr)
            except Exception as e:
                self.sos_kernel.warn('Failed to evaluate {!r}: {}'.format(expr, e))
                return None
        return res

    def sessioninfo(self):
        response = self.sos_kernel.get_response(r'RUBY_VERSION', ('stream',), name=('stdout',))
        return response['text']
