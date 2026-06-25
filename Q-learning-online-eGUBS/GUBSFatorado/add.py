# This file is part of pydd package.

# pydd is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.

# pydd is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# GNU General Public License for more details.

# You should have received a copy of the GNU General Public License
# along with pyddlib. If not, see <http://www.gnu.org/licenses/>.

from numbers import Number
import sys
from dd import DD

class ADD(DD):
	"""
	Reduced Ordered Algebraic Decision Diagram class.

	:param index: root vertex variable index (-1 if terminal vertex)
	:type  index: int
	:param low:   low child vertex of ADD (None if terminal vertex)
	:type  low:   pyddlib.ADD
	:param high:  high child vertex of ADD (None if terminal vertex)
	:type  high:  pyddlib.ADD
	:param value: terminal numeric value (None if non-terminal vertex)
	:type  type:  Number or None
	"""

	__nextid = 1

	def __init__(self, index, low, high, value):
		self._index = index
		self._low   = low
		self._high  = high
		self._value = value
		self._id    = ADD.__nextid
		ADD.__nextid += 1

	def __repr__(self):
		"""
		Return tree-like representation of pyddlib.ADD object.

		:rytpe: str
		"""
		ddrepr = ''
		stack = [(self, 0, None)]
		while stack:
			(vertex, indentation, child_type) = stack.pop()
			for i in range(indentation):
				ddrepr += '|  '
			prefix = '@'
			if child_type is not None:
				prefix = child_type
			ddrepr += prefix
			if vertex.is_terminal():
				ddrepr += ' (value={}, id={})'.format(vertex._value, vertex._id) + '\n'
			else:
				ddrepr += ' (index={}, id={})'.format(vertex._index, vertex._id) + '\n'
				stack.append((vertex._high, indentation+1, '+'))
				stack.append((vertex._low,  indentation+1, '-'))
		return ddrepr

	@property
	def value(self):
		"""
		Return node value.

		:rtype: Number or None
		"""
		return self._value

	@property
	def index(self):
		"""
		Return variable index of node.

		:rtype: int
		"""
		return self._index

	def is_terminal(self):
		"""
		Return True if ADD function represents a constant value.
		Otherwise, return False.

		:rtype: bool
		"""
		return self._low  is None and \
		       self._high is None and \
		       isinstance(self._value, Number)

	def is_constant(self):
		"""
		Return True if ADD function represents a constant value.
		Otherwise, return False.

		:rtype: bool
		"""
		return self.is_terminal()

	def is_variable(self):
		"""
		Return True if ADD function represents the function
		of a single boolean variable. Otherwise, return False.

		:rtype: bool
		"""
		low  = self._low
		high = self._high
		return low  and low.is_terminal()  and low._value  == 0.0 and \
		       high and high.is_terminal() and high._value == 1.0 and \
		       self._value is None

	def __invert__(self):
		"""
		Compute a new reduced ADD representing the negation
		of the algebraic function. Terminal values other than
		0.0 are changed to 0.0 and terminal value 0.0 is changed
		to 1.0.
		Return ~self.

		:rtype: pyddlib.ADD
		"""
		return ADD.reduce(self.__invert_step())

	def __invert_step(self):
		"""
		Return a new ADD representing the negation of the
		algebraic function. Terminal values other than 0.0
		are changed to 0.0 and terminal value 0.0 is changed
		to 1.0.

		:rtype: pyddlib.ADD
		"""
		if self.is_constant():
			if bool(self._value):
				return self.constant(0.0)
			else:
				return self.constant(1.0)
		low  = self._low.__invert_step()
		high = self._high.__invert_step()
		return ADD(self._index, low, high, None)

	def __neg__(self):
		"""
		Compute a new ADD representing the opposite of the
		algebraic function.
		Return -self.

		:rtype: pyddlib.ADD
		"""
		if self.is_constant():
			return self.constant(-1.0 * self._value)
		return ADD(self._index, -self._low, -self._high, None)

	def __add__(self, other):
		"""
		Compute a new ADD representing the addition of algebraic functions.
		Return self+other.

		:param other: ADD
		:type other: pyddlib.ADD
		:rtype: pyddlib.ADD
		"""
		return ADD.apply(self, other, float.__add__)

	def __sub__(self, other):
		"""
		Compute a new ADD representing the subtraction of algebraic functions.
		Return self-other.

		:param other: ADD
		:type other: pyddlib.ADD
		:rtype: pyddlib.ADD
		"""
		return ADD.apply(self, other, float.__sub__)

	def __mul__(self, other):
		"""
		Compute a new ADD representing the product of algebraic functions.
		Return self*other.

		:param other: ADD
		:type other: pyddlib.ADD
		:rtype: pyddlib.ADD
		"""
		return ADD.apply(self, other, float.__mul__)

	def __truediv__(self, other):
		"""
		Compute a new ADD representing the division of algebraic functions.
		Return self/other.

		:param other: ADD
		:type other: pyddlib.ADD
		:rtype: pyddlib.ADD
		"""
		return ADD.apply(self, other, float.__truediv__)

	def __eq__(self, other):
		"""
		Return True if both ADDs represent the same algebraic function.

		:param other: ADD
		:type other: pyddlib.ADD
		:rtype: bool
		"""
		result = ADD.apply(self, other, float.__eq__)
		return result.is_terminal() and bool(result._value)

	def __neq__(self, other):
		"""
		Return True if both ADDs do not represent the same algebraic function.

		:param other: ADD
		:type other: pyddlib.ADD
		:rtype: bool
		"""
		return not self == other

	def marginalize(self, variable):
		"""
		Compute a new reduced ADD with `variable` marginalized.
		Return self.restrict({variable.index: 1}) + self.restrict({variable.index: 0})

		:param variable: ADD variable node
		:type other: pyddlib.ADD
		:rtype: pyddlib.ADD
		"""
		return ADD.reduce(self.__marginalize_step(variable))

	def __marginalize_step(self, variable):
		"""
		Compute a new ADD with `variable` marginalized.

		:param variable: ADD variable node
		:type other: pyddlib.ADD
		:rtype: pyddlib.ADD
		"""
		if self.is_terminal():
			return self
		if self._index == variable._index:
			return self._low + self._high
		low  = self._low.marginalize(variable)
		high = self._high.marginalize(variable)
		return ADD(self._index, low, high, None)

	@classmethod
	def terminal(cls, value):
		"""
		Return a terminal node with a given numeric `value`.

		:param value: numeric value
		:type value: Number
		:rtype: pyddlib.ADD
		"""
		assert(isinstance(value, Number))
		return ADD(-1, None, None, float(value))

	@classmethod
	def constant(cls, value):
		"""
		Return a terminal node with a given numeric `value`.

		:param value: numeric value
		:type value: Number
		:rtype: pyddlib.ADD
		"""
		return cls.terminal(value)

	@classmethod
	def variable(cls, index):
		"""
		Return the ADD representing the function of a
		single boolean variable with given `index`.

		:param index: variable index
		:type index: int
		:rtype: pyddlib.ADD
		"""
		one  = cls.terminal(1.0)
		zero = cls.terminal(0.0)
		return ADD(index, zero, one, None)

#CHDP

	@classmethod
	def variable(cls, index, low, high):
		one = cls.terminal(float(high))
		zero = cls.terminal(float(low))
		return ADD(index, zero, one, None)

	@classmethod
	def subvariable(cls, add, low, high):
		add._low = low
		add._high = high
		return add

	def __abs__(self):
		return ADD.reduce(self.__abs_step())

	def __abs_step(self):
		stack = [self]
		while stack:
			vertex = stack.pop()
			if vertex.is_terminal():
				if vertex._value < 0:
					vertex._value = vertex._value * -1
			else:
				stack.append(vertex._low)
				stack.append(vertex._high)
		return self

	@classmethod
	def max(cls, add):
		higher = 0
		stack = [add]
		while stack:
			vertex= stack.pop()
			if vertex.is_terminal():
				if vertex._value > higher:
					higher = vertex._value
			else:
				stack.append(vertex._low)
				stack.append(vertex._high)
		return higher

	def CONVERTTOPRIMES(self):
		stack = [self]
		while stack:
			vertex = stack.pop()
			if vertex.is_variable():
				vertex._index = vertex._index + 1 #+1 es S'
				stack.append(vertex._low)
				stack.append(vertex._high)

	def getVariables(self):
		var = []
		stack = [self]
		while stack:
			vertex = stack.pop()
			if vertex.is_variable():
				var.append(vertex._index)
				stack.append(vertex._low)
				stack.append(vertex._high)
		return var

	@classmethod
	def CPT(cls, parentstate, s):
		xp = ADD.variable(int(s)+1, 0, 0)#Prime
		if len(parentstate) > 0:
			x1 = ADD.variable(parentstate[0], 0, 0)
			x = x1
			for i in range(1, len(parentstate)):
				x2 = ADD.variable(parentstate[i][0], 0, 0)
				ADD.subvariable(x1, x2, x2)
				x1 = x2
			ADD.subvariable(xp, x, x)
		print(xp)
		return xp
