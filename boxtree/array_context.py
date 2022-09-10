__copyright__ = "Copyright (C) 2022 Alexandru Fikl"

__license__ = """
Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
"""

import numpy as np

from pyopencl.algorithm import BuiltList

from arraycontext import (          # noqa: F401
        PyOpenCLArrayContext as PyOpenCLArrayContextBase,
        with_array_context, serialize_container, deserialize_container,
        rec_map_array_container)
from arraycontext.pytest import (
        _PytestPyOpenCLArrayContextFactoryWithClass,
        register_pytest_array_context_factory)

__doc__ = """
.. autoclass:: PyOpenCLArrayContext
"""


# {{{ array context

class PyOpenCLArrayContext(PyOpenCLArrayContextBase):
    def transform_loopy_program(self, t_unit):
        default_ep = t_unit.default_entrypoint
        options = default_ep.options

        if not (options.return_dict and options.no_numpy):
            raise ValueError("Loopy kernel passed to call_loopy must "
                    "have return_dict and no_numpy options set. "
                    "Did you use arraycontext.make_loopy_program "
                    "to create this kernel?")

        return super().transform_loopy_program(t_unit)

    # NOTE: _rec_map_container is copied from arraycontext wholesale and should
    # be kept in sync as much as possible!

    def _rec_map_container(self, func, array, allowed_types=None, *,
            default_scalar=None, strict=False):
        import arraycontext.impl.pyopencl.taggable_cl_array as tga

        if allowed_types is None:
            allowed_types = (tga.TaggableCLArray,)

        def _wrapper(ary):
            # NOTE: this is copied verbatim from arraycontext and this is the
            # only change to allow optional fields inside containers
            if ary is None:
                return ary

            if isinstance(ary, allowed_types):
                return func(ary)
            elif not strict and isinstance(ary, self.array_types):
                from warnings import warn
                warn(f"Invoking {type(self).__name__}.{func.__name__[1:]} with "
                    f"{type(ary).__name__} will be unsupported in 2023. Use "
                    "'to_tagged_cl_array' to convert instances to TaggableCLArray.",
                    DeprecationWarning, stacklevel=2)
                return func(tga.to_tagged_cl_array(ary))
            elif np.isscalar(ary):
                if default_scalar is None:
                    return ary
                else:
                    return np.array(ary).dtype.type(default_scalar)
            else:
                raise TypeError(
                    f"{type(self).__name__}.{func.__name__[1:]} invoked with "
                    f"an unsupported array type: got '{type(ary).__name__}', "
                    f"but expected one of {allowed_types}")

        return rec_map_array_container(_wrapper, array)

# }}}


# {{{ dataclass array container

def dataclass_array_container(cls: type) -> type:
    """A decorator based on :func:`arraycontext.dataclass_array_container`
    that allows :class:`typing.Optional` containers.
    """

    from dataclasses import Field, fields, is_dataclass
    from typing import Union, get_args
    try:
        # NOTE: only available in python >= 3.8
        from typing import get_origin
    except ImportError:
        from typing_extensions import get_origin

    from arraycontext.container.dataclass import (
        is_array_type, inject_dataclass_serialization)

    assert is_dataclass(cls)

    def is_array_field(f: Field) -> bool:
        if __debug__:
            if not f.init:
                raise ValueError(
                        f"Fields with 'init=False' not allowed: '{f.name}'")

            if isinstance(f.type, str):
                raise TypeError(
                        f"String annotation on field '{f.name}' not supported")

        origin = get_origin(f.type)
        if origin is Union:
            # NOTE: `Optional` is caught in here as an alias for `Union[Anon, type]`
            return all(
                is_array_type(arg) or isinstance(arg, type(None))
                for arg in get_args(f.type))

        from typing import _GenericAlias, _SpecialForm  # type: ignore[attr-defined]
        if isinstance(f.type, (_GenericAlias, _SpecialForm)):
            return False

        return is_array_type(f.type)

    from pytools import partition
    array_fields, non_array_fields = partition(is_array_field, fields(cls))

    if not array_fields:
        raise ValueError(f"'{cls}' must have fields with array container type "
                "in order to use the 'dataclass_array_container' decorator")

    return inject_dataclass_serialization(cls, array_fields, non_array_fields)

# }}}


# {{{ serialization

# NOTE: BuiltList is serialized explicitly here to avoid monkeypatching the
# version in pyopencl (dataclass_array_container modifies the class)

@serialize_container.register(BuiltList)
def _serialize_built_list(obj: BuiltList):
    return tuple([
        ("starts", obj.starts),
        ("lists", obj.lists),
        ("nonempty_indices", obj.nonempty_indices),
        ("compressed_indices", obj.compressed_indices),
        ])


@deserialize_container.register(BuiltList)
def _deserialize_built_list(template: BuiltList, iterable):
    return type(template)(
        count=template.count,
        num_nonempty_lists=template.num_nonempty_lists,
        **dict(iterable))

# }}}


# {{{ pytest

def _acf():
    import pyopencl as cl
    ctx = cl.create_some_context()
    queue = cl.CommandQueue(ctx)

    return PyOpenCLArrayContext(queue, force_device_scalars=True)


class PytestPyOpenCLArrayContextFactory(
        _PytestPyOpenCLArrayContextFactoryWithClass):
    actx_class = PyOpenCLArrayContext


register_pytest_array_context_factory("boxtree.pyopencl",
        PytestPyOpenCLArrayContextFactory)

# }}}