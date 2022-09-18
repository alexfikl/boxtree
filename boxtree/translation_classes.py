"""
Translation classes data structure
----------------------------------

.. autoclass:: TranslationClassesInfo
.. autofunction:: build_translation_classes
"""

__copyright__ = "Copyright (C) 2019 Matt Wala"

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

from functools import partial
from dataclasses import dataclass

import numpy as np
from pyopencl.elementwise import ElementwiseTemplate

from arraycontext import Array
from pytools import memoize_on_first_arg, log_process
from mako.template import Template

from boxtree.tree import Tree
from boxtree.traversal import TRAVERSAL_PREAMBLE_MAKO_DEFS, FMMTraversalInfo
from boxtree.tools import (
    inline_binary_search_for_type, get_coord_vec_dtype, coord_vec_subscript_code)
from boxtree.array_context import PyOpenCLArrayContext, dataclass_array_container

import logging
logger = logging.getLogger(__name__)


# {{{ kernel templates

TRANSLATION_CLASS_FINDER_PREAMBLE_TEMPLATE = Template(r"""//CL:mako//
    #define LEVEL_TO_RAD(level) \
        (root_extent * 1 / (coord_t) (1 << (level + 1)))

    // Return an integer vector indicating the a translation direction
    // as a multiple of the box diameter.
    inline int_coord_vec_t get_normalized_translation_vector(
        coord_t root_extent,
        int level,
        coord_vec_t source_center,
        coord_vec_t target_center)
    {
        int_coord_vec_t result = (int_coord_vec_t) 0;
        coord_t diam = 2 * LEVEL_TO_RAD(level);
        %for i in range(dimensions):
            ${cvec_sub("result", i)} = rint(
                (${cvec_sub("target_center", i)} - ${cvec_sub("source_center", i)})
                / diam);
        %endfor
        return result;
    }

    // Compute the translation class for the given translation vector.  The
    // translation class maps a translation vector (a_1, a_2, ..., a_d) into
    // a dense range of integers [0, ..., (4*n+3)^d - 1], where
    // d is the dimension and n is well_sep_is_n_away.
    //
    // The translation vector should be normalized for a box diameter of 1.
    //
    // This relies on the fact that the entries of the vector will
    // always be in the range [-2n-1,...,2n+1].
    //
    // The mapping from vector to class is:
    //
    //                         \~~   d                 k-1
    //     cls(a ,a ,...,a ) =  >      (2n+1+a ) (4n+3)
    //          1  2      d    /__ k=1        k
    //
    // Returns -1 on error.
    inline int get_translation_class(int_coord_vec_t vec, int well_sep_is_n_away)
    {
        int dim_bound = 2 * well_sep_is_n_away + 1;
        %for i in range(dimensions):
            if (!(-dim_bound <= ${cvec_sub("vec", i)}
                && ${cvec_sub("vec", i)} <= dim_bound))
            {
                return -1;
            }
        %endfor

        int result = 0;
        int base = 4 * well_sep_is_n_away + 3;
        int mult = 1;
        %for i in range(dimensions):
            result += (2 * well_sep_is_n_away + 1 + ${cvec_sub("vec", i)}) * mult;
            mult *= base;
        %endfor
        return result;
    }
    """ + inline_binary_search_for_type("box_id_t"),
    strict_undefined=True)


TRANSLATION_CLASS_FINDER_TEMPLATE = ElementwiseTemplate(
    arguments=r"""//CL:mako//
    /* input: */
    box_id_t *from_sep_siblings_lists,
    box_id_t *from_sep_siblings_starts,
    box_id_t *target_or_target_parent_boxes,
    int ntarget_or_target_parent_boxes,
    coord_t *box_centers,
    int aligned_nboxes,
    coord_t root_extent,
    box_level_t *box_levels,
    int well_sep_is_n_away,

    /* output: */
    int *translation_classes,
    int *translation_class_is_used,
    int *error_flag,
    """,

    operation=TRAVERSAL_PREAMBLE_MAKO_DEFS + r"""//CL:mako//
    // Find the target box for this source box.
    box_id_t source_box_id = from_sep_siblings_lists[i];

    size_t itarget_box = bsearch(
        from_sep_siblings_starts, 1 + ntarget_or_target_parent_boxes, i);

    box_id_t target_box_id = target_or_target_parent_boxes[itarget_box];

    // Ensure levels are the same.
    if (box_levels[source_box_id] != box_levels[target_box_id])
    {
        atomic_or(error_flag, 1);
        PYOPENCL_ELWISE_CONTINUE;
    }

    // Compute the translation vector and translation class.
    ${load_center("source_center", "source_box_id")}
    ${load_center("target_center", "target_box_id")}

    int_coord_vec_t vec = get_normalized_translation_vector(
        root_extent, box_levels[source_box_id], source_center, target_center);

    int translation_class = get_translation_class(vec, well_sep_is_n_away);

    // Ensure valid translation class.
    if (translation_class == -1)
    {
        atomic_or(error_flag, 1);
        PYOPENCL_ELWISE_CONTINUE;
    }

    % if translation_class_per_level:
        translation_class += box_levels[source_box_id] * \
                                ${ntranslation_classes_per_level};
    % endif

    translation_classes[i] = translation_class;
    atomic_or(&translation_class_is_used[translation_class], 1);
    """)

# }}}


# {{{ translation classes builder

class TranslationClassesBuilder:
    def __init__(self, *args, **kargs):
        pass

    def __call__(self, actx: PyOpenCLArrayContext,
            trav, tree, wait_for=None, is_translation_per_level=True):
        from warnings import warn
        warn(f"'{type(self).__name__}' is deprecated and will be removed in 2023. "
            "Use 'build_rotation_classes' instead.",
            DeprecationWarning, stacklevel=2)

        return build_translation_classes(
            actx, trav, tree,
            is_translation_per_level=is_translation_per_level)


@dataclass_array_container
@dataclass(frozen=True)
class TranslationClassesInfo:
    r"""Interaction lists to help with for translations that benefit from
    precomputing distance related values

    .. attribute:: nfrom_sep_siblings_translation_classes

       The number of distinct translation classes.

    .. attribute:: from_sep_siblings_translation_classes

        ``int32 [*]``

        A list, corresponding to *from_sep_siblings_lists* of :attr:`traversal`, of
        the translation classes of each box pair.

    .. attribute:: from_sep_siblings_translation_class_to_distance_vector

        ``coord_vec_t [nfrom_sep_siblings_translation_classes]``

        Maps translation classes in *from_sep_siblings_translation_classes*
        to distance (translation) vectors from source box center to
        target box center.

    .. attribute:: from_sep_siblings_translation_classes_level_starts

        ``int32 [nlevels + 1]``

        A list with an entry for each level giving the starting translation
        class id for that level. Translation classes are numbered contiguously
        by level.

    .. attribute:: traversal

        A :class:`boxtree.traversal.FMMTraversalInfo` object corresponding to the
        traversal that these translation classes refer to.
    """

    traversal: FMMTraversalInfo
    from_sep_siblings_translation_classes: Array
    from_sep_siblings_translation_class_to_distance_vector: Array
    from_sep_siblings_translation_classes_level_starts: Array

    @property
    def nfrom_sep_siblings_translation_classes(self):
        return len(self.from_sep_siblings_translation_class_to_distance_vector)


def ntranslation_classes_per_level(well_sep_is_n_away: int, dimensions: int) -> int:
    return (4 * well_sep_is_n_away + 3) ** dimensions


def translation_class_to_normalized_vector(
        well_sep_is_n_away: int, dimensions: int, nclasses: int
        ) -> np.ndarray:
    # This computes the vector for the translation class, using the inverse
    # of the formula found in get_translation_class() defined in
    # TRANSLATION_CLASS_FINDER_PREAMBLE_TEMPLATE.
    assert 0 <= nclasses < ntranslation_classes_per_level(
        well_sep_is_n_away, dimensions)

    result = np.zeros(dimensions, dtype=np.int32)
    shift = 2 * well_sep_is_n_away + 1
    base = 4 * well_sep_is_n_away + 3
    for i in range(dimensions):
        result[i] = nclasses % base - shift
        nclasses //= base

    return result


@memoize_on_first_arg
def get_translation_class_finder_kernel(
        actx: PyOpenCLArrayContext,
        dimensions: int,
        well_sep_is_n_away: int,
        box_id_dtype: "np.dtype",
        box_level_dtype: "np.dtype",
        coord_dtype: "np.dtype",
        is_translation_per_level: bool
        ):
    coord_vec_dtype = get_coord_vec_dtype(coord_dtype, dimensions)
    int_coord_vec_dtype = get_coord_vec_dtype(np.dtype(np.int32), dimensions)

    num_translation_classes = (
        ntranslation_classes_per_level(well_sep_is_n_away, dimensions))

    # Make sure translation classes can fit inside a 32 bit integer.
    if not num_translation_classes <= 1 + np.iinfo(np.int32).max:
        raise ValueError("would overflow")

    preamble = TRANSLATION_CLASS_FINDER_PREAMBLE_TEMPLATE.render(
            dimensions=dimensions,
            cvec_sub=partial(coord_vec_subscript_code, dimensions))

    return TRANSLATION_CLASS_FINDER_TEMPLATE.build(
        actx.context,
        type_aliases=(
            ("int_coord_vec_t", int_coord_vec_dtype),
            ("coord_vec_t", coord_vec_dtype),
            ("coord_t", coord_dtype),
            ("box_id_t", box_id_dtype),
            ("box_level_t", box_level_dtype),
        ),
        var_values=(
            ("dimensions", dimensions),
            ("ntranslation_classes_per_level", num_translation_classes),
            ("translation_class_per_level", is_translation_per_level),
            ("cvec_sub", partial(
                coord_vec_subscript_code, dimensions)),
        ),
        more_preamble=preamble)


def compute_used_translation_classes(
        actx: PyOpenCLArrayContext, trav: FMMTraversalInfo, tree: Tree, *,
        is_translation_per_level: bool):
    # {{{ compute translation classes for list 2

    well_sep_is_n_away = trav.well_sep_is_n_away
    dimensions = tree.dimensions
    coord_dtype = tree.coord_dtype

    ntranslation_classes = (
        ntranslation_classes_per_level(well_sep_is_n_away, dimensions))

    if is_translation_per_level:
        ntranslation_classes = ntranslation_classes * tree.nlevels

    translation_classes_lists = actx.empty(
        len(trav.from_sep_siblings_lists), dtype=np.int32)
    translation_class_is_used = actx.zeros(ntranslation_classes, dtype=np.int32)
    error_flag = actx.zeros(1, dtype=np.int32)

    translation_class_finder_knl = get_translation_class_finder_kernel(
        actx,
        dimensions, well_sep_is_n_away,
        tree.box_id_dtype, tree.box_level_dtype, coord_dtype,
        is_translation_per_level,
        )

    evt = translation_class_finder_knl(
            trav.from_sep_siblings_lists,
            trav.from_sep_siblings_starts,
            trav.target_or_target_parent_boxes,
            trav.ntarget_or_target_parent_boxes,
            tree.box_centers,
            tree.aligned_nboxes,
            tree.root_extent,
            tree.box_levels,
            well_sep_is_n_away,
            translation_classes_lists,
            translation_class_is_used,
            error_flag,
            queue=actx.queue,
            )
    translation_classes_lists.add_event(evt)
    translation_class_is_used.add_event(evt)

    if actx.to_numpy(error_flag):
        raise ValueError("could not compute translation classes")

    return translation_class_is_used, translation_classes_lists

    # }}}


@log_process(logger, "build m2l translation classes")
def build_translation_classes(actx: PyOpenCLArrayContext,
        trav: FMMTraversalInfo, tree: Tree, *,
        is_translation_per_level: bool = True) -> TranslationClassesInfo:
    """Build translation classes for List 2 translations."""
    translation_class_is_used, translation_classes_lists = (
        compute_used_translation_classes(actx, trav, tree,
            is_translation_per_level=is_translation_per_level))

    well_sep_is_n_away = trav.well_sep_is_n_away
    dimensions = tree.dimensions

    used_translation_classes_map = np.empty(
        len(translation_class_is_used), dtype=np.int32)
    used_translation_classes_map.fill(-1)

    distances = np.empty(
        (dimensions, len(translation_class_is_used)), dtype=tree.coord_dtype)
    num_translation_classes = (
        ntranslation_classes_per_level(well_sep_is_n_away, dimensions))

    nlevels = tree.nlevels
    count = 0
    prev_level = -1
    from_sep_siblings_translation_classes_level_starts = (
        np.empty(nlevels + 1, dtype=np.int32))

    for i, used in enumerate(actx.to_numpy(translation_class_is_used)):
        cls_without_level = i % num_translation_classes
        level = i // num_translation_classes
        if (prev_level != level):
            from_sep_siblings_translation_classes_level_starts[level] = count
            prev_level = level

        if not used:
            continue

        used_translation_classes_map[i] = count
        unit_vector = translation_class_to_normalized_vector(
            well_sep_is_n_away, dimensions, cls_without_level)

        distances[:, count] = unit_vector * tree.root_extent / (1 << level)
        count = count + 1

    from_sep_siblings_translation_classes_level_starts[nlevels] = count

    translation_classes_lists = actx.from_numpy(
        used_translation_classes_map
        )[translation_classes_lists]

    distances = actx.from_numpy(distances)
    from_sep_siblings_translation_classes_level_starts = actx.from_numpy(
        from_sep_siblings_translation_classes_level_starts)

    info = TranslationClassesInfo(
            traversal=trav,
            from_sep_siblings_translation_classes=translation_classes_lists,
            from_sep_siblings_translation_class_to_distance_vector=distances,
            from_sep_siblings_translation_classes_level_starts=(
                from_sep_siblings_translation_classes_level_starts),
            )

    return actx.freeze(info)

# }}}

# vim: fdm=marker
