#!/usr/bin/env python3
#
# Copyright (c) Bo Peng and the University of Texas MD Anderson Cancer Center
# Distributed under the terms of the 3-clause BSD License.

import json
from collections.abc import Sequence

import numpy
import pandas
from sos.utils import env, short_repr

Ruby_init_statement = r'''

require 'daru'

require 'nmatrix'

def __Ruby_py_repr(obj)
  if obj.is_a? Integer
    return obj.inspect
  elsif obj.is_a? String
    return obj.inspect
  elsif obj.is_a? TrueClass
    return "True"
  elsif obj.is_a? FalseClass
    return "False"
  elsif obj.is_a? Float
    return obj.inspect
  elsif obj.nil?
    return "None"
  elsif obj.is_a? Set
    return "{" + (obj.map { |indivial_var| __Ruby_py_repr(indivial_var) } ).join(",") + "}"
  elsif obj.is_a? Range
    return "range(" + obj.min().inspect + "," + (obj.max()+1).inspect + ")"
  elsif obj.is_a? Array
    return '[' + (obj.map { |indivial_var| __Ruby_py_repr(indivial_var) } ).join(",") + ']'
  elsif obj.is_a? Hash
    _beginning_result_string_hash_from_ruby = "{"
    _context_result_string_hash_from_ruby = (obj.keys.map do |x|
                                              if obj[x].is_a? Array then
                                                  "\"" + x.to_s + "\":" + (obj[x].to_a.map { |y|  eval(__Ruby_py_repr(y)) }).to_s
                                              else
                                                  "\"" + x.to_s + "\":" + (__Ruby_py_repr(obj[x])).to_s
                                              end
                                            end).join(",") + "}"
    _result_string_hash_from_ruby = _beginning_result_string_hash_from_ruby + _context_result_string_hash_from_ruby
    return _result_string_hash_from_ruby
  elsif obj.is_a? Daru::DataFrame
    _beginning_result_string_dataframe_from_ruby = "pandas.DataFrame(" + "{"
    _context_result_string_dataframe_from_ruby = (obj.vectors.to_a.map { |x| "\"" + x.to_s + "\":" + (obj[x].to_a.map { |y|  eval(__Ruby_py_repr(y)) }).to_s } ).join(",")
    _indexing_result_string_dataframe_from_ruby = "}," + "index=" + obj.index.to_a.to_s + ")"
    _result_string_dataframe_from_ruby = _beginning_result_string_dataframe_from_ruby + _context_result_string_dataframe_from_ruby + _indexing_result_string_dataframe_from_ruby
    return _result_string_dataframe_from_ruby
  elsif obj.is_a? NMatrix
    return "numpy.matrix(" + obj.to_a.to_s + ")"
  elsif obj.is_a? Complex
    return "complex(" + obj.real.inspect + "," + obj.imaginary.inspect + ")"
  else
    return "'Untransferrable variable'"
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
        if isinstance(obj, float) and numpy.isnan(obj):
            return "Float::NAN"
        if isinstance(obj, (int, float)):
            return repr(obj)
        if isinstance(obj, str):
            return '%(' + obj + ')'
        if isinstance(obj, complex):
            return 'Complex(' + str(obj.real) + ',' + str(obj.imag) + ')'
        if isinstance(obj, range):
            return '(' + repr(min(obj)) + '...' + repr(max(obj)) + ')'
        if isinstance(obj, Sequence):
            if len(obj) == 0:
                return '[]'
            return '[' + ','.join(self._Ruby_repr(x) for x in obj) + ']'
        if obj is None:
            return 'nil'
        if isinstance(obj, dict):
            return '{' + ','.join(f'"{x}" => {self._Ruby_repr(y)}' for x, y in obj.items()) + '}'
        if isinstance(obj, set):
            return 'Set[' + ','.join(self._Ruby_repr(x) for x in obj) + ']'
        if isinstance(obj, (numpy.intc, numpy.intp, numpy.int8, numpy.int16, numpy.int32, numpy.int64,\
                numpy.uint8, numpy.uint16, numpy.uint32, numpy.uint64, numpy.float16, numpy.float32, numpy.float64)):
            return repr(obj)
        if isinstance(obj, numpy.matrixlib.defmatrix.matrix):
            return 'N' + repr(obj.tolist())
        if isinstance(obj, numpy.ndarray):
            return repr(obj.tolist())
        if isinstance(obj, pandas.DataFrame):
            _beginning_result_string_dataframe_to_ruby = "Daru::DataFrame.new({"
            _context_string_dataframe_to_ruby = str(['"'
                                                    + str(x).replace("'", '"')
                                                    + '"'
                                                    + "=>"
                                                    + "["
                                                    + str(
                                                        ",".join(
                                                            list(
                                                            map(
                                                                lambda y: self._Ruby_repr(y),
                                                                    obj[x].tolist()
                                                            )
                                                        )
                                                        )
                                                    ).replace("'", '"') + "]"
                                                    for x in obj.keys().tolist()])[2:-2].replace("\', \'", ", ") + "},"
            _indexing_result_string_dataframe_to_ruby = "index:" + str(obj.index.values.tolist()).replace("'", '"') + ")"
            _result_string_dataframe_to_ruby = _beginning_result_string_dataframe_to_ruby + _context_string_dataframe_to_ruby + _indexing_result_string_dataframe_to_ruby
            return _result_string_dataframe_to_ruby
        if isinstance(obj, pandas.Series):
            dat=list(obj.values)
            ind=list(obj.index.values)
            ans="{" + ",".join([repr(x) + "=>" + repr(y) for x, y in zip(ind, dat)]) + "}"
            return ans
        return repr(f'Unsupported datatype {short_repr(obj)}')

    async def get_vars(self, names, as_var=None):
        for name in names:
            newname = as_var if as_var else name
            ruby_repr = self._Ruby_repr(env.sos_dict[name])
            await self.sos_kernel.run_cell(f'{newname} = {ruby_repr}', True, False,
                                     on_error=f'Failed to put variable {name} to Ruby')

    def put_vars(self, items, to_kernel=None, as_var=None):
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
            py_repr = f'print(__Ruby_py_repr({item}))'
            response = self.sos_kernel.get_response(py_repr, ('stream',), name=('stdout',))[0][1]
            expr = response['text']
            self.sos_kernel.warn(repr(expr))

            try:
                # evaluate as raw string to correctly handle \\ etc
                res[as_var if as_var else item] = eval(expr)
            except Exception as e:
                self.sos_kernel.warn(f'Failed to evaluate {expr!r}: {e}')
                return None
        return res

    def sessioninfo(self):
        response = self.sos_kernel.get_response(r'RUBY_VERSION', ('stream',), name=('stdout',))
        return response['text']
