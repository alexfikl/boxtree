__copyright__ = """
Copyright (C) 2013 Andreas Kloeckner
Copyright (C) 2016 Matt Wala"""

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
from pytools import ProcessLogger, memoize_on_first_arg
from mako.template import Template

from boxtree.tree import Tree
from boxtree.tools import (
    inline_binary_search_for_type, get_coord_vec_dtype, coord_vec_subscript_code)
from boxtree.array_context import PyOpenCLArrayContext, dataclass_array_container

import logging
logger = logging.getLogger(__name__)


__doc__ = """
Area queries (Balls -> overlapping leaves)
------------------------------------------

.. autoclass:: AreaQueryResult
.. autofunction:: build_area_query


Inverse of area query (Leaves -> overlapping balls)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: LeavesToBallsLookup
.. autofunction:: build_leaves_to_balls_lookup

Space invader queries
^^^^^^^^^^^^^^^^^^^^^

.. autofunction:: build_space_invader_query

Peer Lists
^^^^^^^^^^

Area queries are implemented using peer lists.

.. autoclass:: PeerListLookup
.. autofunction:: build_peer_list
"""


# {{{ kernel templates

GUIDING_BOX_FINDER_MACRO = r"""//CL:mako//
    <%def name="initialize_coord_vec(vector_name, entries)">
        <% assert len(entries) == dimensions %>
        ${vector_name} = (coord_vec_t) (${", ".join(entries)});
    </%def>

    <%def name="find_guiding_box(ball_center, ball_radius, box='guiding_box')">
        box_id_t ${box} = 0;
        {
            //
            // Step 1: Ensure that the center is within the bounding box.
            //
            coord_vec_t query_center, bbox_min, bbox_max;

            ${initialize_coord_vec(
                "bbox_min", ["bbox_min_" + ax for ax in AXIS_NAMES[:dimensions]])}

            // bbox_max should be smaller than the true bounding box, so that
            // (query_center - bbox_min) / root_extent entries are in the half open
            // interval [0, 1).
            bbox_max = bbox_min + (coord_t) (
                root_extent / (1 + ${root_extent_stretch_factor}));
            query_center = min(bbox_max, max(bbox_min, ${ball_center}));

            //
            // Step 2: Compute the query radius. This can be effectively
            // smaller than the original ball radius, if the center
            // isn't in the bounding box (see picture):
            //
            //          +-----------+
            //          |           |
            //          |           |
            //          |           |
            //  +-------|-+         |
            //  |       | |         |
            //  |       +-----------+ <= bounding box
            //  |         |
            //  |         |
            //  +---------+ <= query box
            //
            //        <---> <= original radius
            //           <> <= effective radius
            //
            coord_t query_radius = 0;

            %for mnr in range(2**dimensions):
            {
                coord_vec_t offset;

                ${initialize_coord_vec("offset",
                    ["{sign}{ball_radius}".format(
                        sign="+" if (2**(dimensions-1-idim) & mnr) else "-",
                        ball_radius=ball_radius)
                    for idim in range(dimensions)])}

                coord_vec_t corner = min(
                    bbox_max, max(bbox_min, ${ball_center} + offset));

                coord_vec_t dist = fabs(corner - query_center);
                %for i in range(dimensions):
                    query_radius = fmax(query_radius, ${cvec_sub("dist", i)});
                %endfor
            }
            %endfor

            //
            // Step 3: Find the guiding box.
            //

            // Descend when root is not the guiding box.
            if (LEVEL_TO_RAD(0) / 2 >= query_radius)
            {
                for (unsigned box_level = 0;; ++box_level)
                {
                    if (/* Found leaf? */
                        !(box_flags[${box}] & BOX_HAS_CHILDREN)
                        /* Found guiding box? */
                        || (LEVEL_TO_RAD(box_level) / 2 < query_radius
                            && query_radius <= LEVEL_TO_RAD(box_level)))
                    {
                        break;
                    }

                    // Find the child containing the ball center.
                    //
                    // Logic intended to match the morton nr scan kernel.

                    coord_vec_t offset_scaled =
                        (query_center - bbox_min) / root_extent;

                    // Invariant: offset_scaled entries are in [0, 1).
                    %for ax in AXIS_NAMES[:dimensions]:
                        unsigned ${ax}_bits = (unsigned) (
                            offset_scaled.${ax} * (1U << (1 + box_level)));
                    %endfor

                    // Pick off the lowest-order bit for each axis, put it in
                    // its place.
                    int level_morton_number = 0
                    %for iax, ax in enumerate(AXIS_NAMES[:dimensions]):
                        | (${ax}_bits & 1U) << (${dimensions-1-iax})
                    %endfor
                        ;

                    box_id_t next_box = box_child_ids[
                        level_morton_number * aligned_nboxes + ${box}];

                    if (next_box)
                    {
                        ${box} = next_box;
                    }
                    else
                    {
                        // Child does not exist, this must be the guiding box
                        break;
                    }
                }
            }
        }
    </%def>
"""


AREA_QUERY_WALKER_BODY = r"""
    coord_vec_t ball_center;
    coord_t ball_radius;
    ${get_ball_center_and_radius("ball_center", "ball_radius", "i")}

    ///////////////////////////////////
    // Step 1: Find the guiding box. //
    ///////////////////////////////////

    ${find_guiding_box("ball_center", "ball_radius")}

    //////////////////////////////////////////////////////
    // Step 2 - Walk the peer boxes to find the leaves. //
    //////////////////////////////////////////////////////

    for (peer_list_idx_t pb_i = peer_list_starts[guiding_box],
         pb_e = peer_list_starts[guiding_box+1]; pb_i < pb_e; ++pb_i)
    {
        box_id_t peer_box = peer_lists[pb_i];

        if (!(box_flags[peer_box] & BOX_HAS_CHILDREN))
        {
            bool is_overlapping;

            ${check_l_infty_ball_overlap(
                "is_overlapping", "peer_box", "ball_radius", "ball_center")}

            if (is_overlapping)
            {
                ${leaf_found_op("peer_box", "ball_center", "ball_radius")}
            }
        }
        else
        {
            ${walk_init("peer_box")}

            while (continue_walk)
            {
                ${walk_get_box_id()}

                if (walk_box_id)
                {
                    if (!(box_flags[walk_box_id] & BOX_HAS_CHILDREN))
                    {
                        bool is_overlapping;

                        ${check_l_infty_ball_overlap(
                            "is_overlapping", "walk_box_id",
                            "ball_radius", "ball_center")}

                        if (is_overlapping)
                        {
                            ${leaf_found_op(
                                "walk_box_id", "ball_center", "ball_radius")}
                        }
                    }
                    else
                    {
                        // We want to descend into this box. Put the current state
                        // on the stack.
                        ${walk_push("walk_box_id")}
                        continue;
                    }
                }

                ${walk_advance()}
            }
        }
    }
"""


AREA_QUERY_TEMPLATE = (
    GUIDING_BOX_FINDER_MACRO + r"""//CL//
    typedef ${dtype_to_ctype(ball_id_dtype)} ball_id_t;
    typedef ${dtype_to_ctype(peer_list_idx_dtype)} peer_list_idx_t;

    <%def name="get_ball_center_and_radius(ball_center, ball_radius, i)">
        %for ax in AXIS_NAMES[:dimensions]:
            ${ball_center}.${ax} = ball_${ax}[${i}];
        %endfor
       ${ball_radius} = ball_radii[${i}];
    </%def>

    <%def name="leaf_found_op(leaf_box_id, ball_center, ball_radius)">
        APPEND_leaves(${leaf_box_id});
    </%def>

    void generate(LIST_ARG_DECL USER_ARG_DECL ball_id_t i)
    {
    """
    + AREA_QUERY_WALKER_BODY
    + """
    }
    """)


PEER_LIST_FINDER_TEMPLATE = r"""//CL//

void generate(LIST_ARG_DECL USER_ARG_DECL box_id_t box_id)
{
    ${load_center("center", "box_id")}

    if (box_id == 0)
    {
        // Peer of root = self
        APPEND_peers(box_id);
        return;
    }

    int level = box_levels[box_id];

    // To find this box's peers, start at the top of the tree, descend
    // into adjacent (or overlapping) parents.
    ${walk_init(0)}

    while (continue_walk)
    {
        ${walk_get_box_id()}

        if (walk_box_id)
        {
            ${load_center("walk_center", "walk_box_id")}

            // walk_box_id lives on level walk_stack_size+1.
            bool a_or_o = is_adjacent_or_overlapping(root_extent,
                center, level, walk_center, walk_stack_size+1);

            if (a_or_o)
            {
                // walk_box_id lives on level walk_stack_size+1.
                if (walk_stack_size+1 == level)
                {
                    APPEND_peers(walk_box_id);
                }
                else if (!(box_flags[walk_box_id] & BOX_HAS_CHILDREN))
                {
                    APPEND_peers(walk_box_id);
                }
                else
                {
                    // Check if any children are adjacent or overlapping.
                    // If not, this box must be a peer.
                    bool must_be_peer = true;

                    for (int morton_nr = 0;
                         must_be_peer && morton_nr < ${2**dimensions};
                         ++morton_nr)
                    {
                        box_id_t next_child_id = box_child_ids[
                            morton_nr * aligned_nboxes + walk_box_id];
                        if (next_child_id)
                        {
                            ${load_center("next_walk_center", "next_child_id")}
                            must_be_peer &= !is_adjacent_or_overlapping(root_extent,
                                center, level, next_walk_center, walk_stack_size+2);
                        }
                    }

                    if (must_be_peer)
                    {
                        APPEND_peers(walk_box_id);
                    }
                    else
                    {
                        // We want to descend into this box. Put the current state
                        // on the stack.
                        ${walk_push("walk_box_id")}
                        continue;
                    }
                }
            }
        }

        ${walk_advance()}
    }
}

"""

STARTS_EXPANDER_TEMPLATE = ElementwiseTemplate(
    arguments=r"""
        idx_t *dst,
        idx_t *starts,
        idx_t starts_len
    """,
    operation=r"""//CL//
    /* Find my index in starts, place the index in dst. */
    dst[i] = bsearch(starts, starts_len, i);
    """,
    name="starts_expander",
    preamble=inline_binary_search_for_type("idx_t"))

# }}}


# {{{ area query elementwise template

class AreaQueryElementwiseTemplate:
    """
    Experimental: Intended as a way to perform operations in the body of an area
    query.
    """

    @staticmethod
    def unwrap_args(tree, peer_lists, *args):
        return (tree.box_centers,
                tree.root_extent,
                tree.box_levels,
                tree.aligned_nboxes,
                tree.box_child_ids,
                tree.box_flags,
                peer_lists.peer_list_starts,
                peer_lists.peer_lists) + tuple(tree.bounding_box[0]) + args

    def __init__(self, extra_args, ball_center_and_radius_expr,
                 leaf_found_op, preamble="", name="area_query_elwise"):

        def wrap_in_macro(decl, expr):
            return """
            <%def name=\"{decl}\">
            {expr}
            </%def>
            """.format(decl=decl, expr=expr)

        from boxtree.traversal import TRAVERSAL_PREAMBLE_MAKO_DEFS

        self.elwise_template = ElementwiseTemplate(
            arguments=r"""//CL:mako//
                coord_t *box_centers,
                coord_t root_extent,
                box_level_t *box_levels,
                box_id_t aligned_nboxes,
                box_id_t *box_child_ids,
                box_flags_t *box_flags,
                peer_list_idx_t *peer_list_starts,
                box_id_t *peer_lists,
                %for ax in AXIS_NAMES[:dimensions]:
                    coord_t bbox_min_${ax},
                %endfor
            """ + extra_args,
            operation="//CL:mako//\n"
            + wrap_in_macro(
                "get_ball_center_and_radius(ball_center, ball_radius, i)",
                ball_center_and_radius_expr)
            + wrap_in_macro(
                "leaf_found_op(leaf_box_id, ball_center, ball_radius)",
                leaf_found_op)
            + TRAVERSAL_PREAMBLE_MAKO_DEFS
            + GUIDING_BOX_FINDER_MACRO
            + AREA_QUERY_WALKER_BODY,
            name=name,
            preamble=preamble)

    def generate(self, context,
                 dimensions, coord_dtype, box_id_dtype,
                 peer_list_idx_dtype, max_levels,
                 extra_var_values=(), extra_type_aliases=(),
                 extra_preamble="",
                 root_extent_stretch_factor=1.0e-4):
        from pyopencl.tools import dtype_to_ctype

        from boxtree import box_flags_enum
        from boxtree.tools import AXIS_NAMES
        from boxtree.traversal import TRAVERSAL_PREAMBLE_TYPEDEFS_AND_DEFINES

        from pyopencl.cltypes import vec_types
        render_vars = (
            ("np", np),
            ("dimensions", dimensions),
            ("dtype_to_ctype", dtype_to_ctype),
            ("box_id_dtype", box_id_dtype),
            ("particle_id_dtype", None),
            ("coord_dtype", coord_dtype),
            ("get_coord_vec_dtype", get_coord_vec_dtype),
            ("cvec_sub", partial(coord_vec_subscript_code, dimensions)),
            ("max_levels", max_levels),
            ("AXIS_NAMES", AXIS_NAMES),
            ("box_flags_enum", box_flags_enum),
            ("peer_list_idx_dtype", peer_list_idx_dtype),
            ("debug", False),
            ("root_extent_stretch_factor", root_extent_stretch_factor),

            # FIXME This gets used in pytential with a template that still uses this:
            ("vec_types", tuple(vec_types.items())),

        )

        preamble = Template(
            # HACK: box_flags_t and coord_t are defined here and
            # in the template below, so disable typedef redefinition warnings.
            """
            #pragma clang diagnostic push
            #pragma clang diagnostic ignored "-Wtypedef-redefinition"
            """
            + TRAVERSAL_PREAMBLE_TYPEDEFS_AND_DEFINES
            + """
            #pragma clang diagnostic pop
            """,
            strict_undefined=True).render(**dict(render_vars))

        return self.elwise_template.build(context,
                type_aliases=(
                    ("coord_t", coord_dtype),
                    ("box_id_t", box_id_dtype),
                    ("peer_list_idx_t", peer_list_idx_dtype),
                    ("box_level_t", np.uint8),
                    ("box_flags_t", box_flags_enum.dtype),
                ) + extra_type_aliases,
                var_values=render_vars + extra_var_values,
                more_preamble=preamble + extra_preamble)


SPACE_INVADER_QUERY_TEMPLATE = AreaQueryElementwiseTemplate(
    extra_args="""
    coord_t *ball_radii,
    float *outer_space_invader_dists,
    %for ax in AXIS_NAMES[:dimensions]:
        coord_t *ball_${ax},
    %endfor
    """,
    ball_center_and_radius_expr=r"""
    ${ball_radius} = ball_radii[${i}];
    %for ax in AXIS_NAMES[:dimensions]:
        ${ball_center}.${ax} = ball_${ax}[${i}];
    %endfor
    """,
    leaf_found_op=r"""
    {
        ${load_center("leaf_center", leaf_box_id)}

        coord_t max_dist = 0;
        %for i in range(dimensions):
            max_dist = fmax(max_dist,
                distance(
                    ${cvec_sub(ball_center, i)},
                    ${cvec_sub("leaf_center", i)}));
        %endfor

        // The atomic max operation supports only integer types.
        // However, max_dist is of a floating point type.
        // For comparison purposes we reinterpret the bits of max_dist
        // as an integer. The comparison result is the same as for positive
        // IEEE floating point numbers, so long as the float/int endianness
        // matches (fingers crossed).
        atomic_max(
            (volatile __global int *)
                &outer_space_invader_dists[${leaf_box_id}],
            as_int((float) max_dist));
    }""",
    name="space_invader_query")

# }}}


# {{{ area query build

class AreaQueryBuilder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, actx: PyOpenCLArrayContext, tree: Tree,
                 ball_centers, ball_radii, peer_lists=None,
                 wait_for=None):
        from warnings import warn
        warn(f"'{type(self).__name__}' is deprecated and will be removed in 2023. "
            "Use 'build_area_query' instead.",
            DeprecationWarning, stacklevel=2)

        return build_area_query(
            actx, tree, ball_centers, ball_radii, peer_lists)


@dataclass_array_container
@dataclass(frozen=True)
class AreaQueryResult:
    """
    .. attribute:: tree

        The :class:`boxtree.Tree` instance used to build this lookup.

    .. attribute:: leaves_near_ball_starts

        Indices into :attr:`leaves_near_ball_lists`.
        ``leaves_near_ball_lists[leaves_near_ball_starts[ball_nr]:
        leaves_near_ball_starts[ball_nr]+1]``
        results in a list of leaf boxes that intersect `ball_nr`.

    .. attribute:: leaves_near_ball_lists

    .. versionadded:: 2016.1
    """

    tree: Tree
    leaves_near_ball_starts: Array
    leaves_near_ball_lists: Array


@memoize_on_first_arg
def get_area_query_kernel(
        actx: PyOpenCLArrayContext,
        dimensions: int,
        coord_dtype: "np.dtype",
        box_id_dtype: "np.dtype",
        ball_id_dtype: "np.dtype",
        peer_list_idx_dtype: "np.dtype",
        max_levels: int,
        root_extent_stretch_factor: float):
    from pyopencl.tools import dtype_to_ctype

    from boxtree import box_flags_enum
    from boxtree.tools import AXIS_NAMES
    from boxtree.traversal import TRAVERSAL_PREAMBLE_TEMPLATE

    logger.debug("start building area query kernel")

    template = Template(
        TRAVERSAL_PREAMBLE_TEMPLATE
        + AREA_QUERY_TEMPLATE,
        strict_undefined=True)

    render_vars = dict(
        np=np,
        dimensions=dimensions,
        dtype_to_ctype=dtype_to_ctype,
        box_id_dtype=box_id_dtype,
        particle_id_dtype=None,
        coord_dtype=coord_dtype,
        get_coord_vec_dtype=get_coord_vec_dtype,
        cvec_sub=partial(coord_vec_subscript_code, dimensions),
        max_levels=max_levels,
        AXIS_NAMES=AXIS_NAMES,
        box_flags_enum=box_flags_enum,
        peer_list_idx_dtype=peer_list_idx_dtype,
        ball_id_dtype=ball_id_dtype,
        debug=False,
        root_extent_stretch_factor=root_extent_stretch_factor)

    from boxtree.tools import VectorArg, ScalarArg
    arg_decls = [
        VectorArg(coord_dtype, "box_centers", with_offset=False),
        ScalarArg(coord_dtype, "root_extent"),
        VectorArg(np.uint8, "box_levels"),
        ScalarArg(box_id_dtype, "aligned_nboxes"),
        VectorArg(box_id_dtype, "box_child_ids", with_offset=False),
        VectorArg(box_flags_enum.dtype, "box_flags"),
        VectorArg(peer_list_idx_dtype, "peer_list_starts"),
        VectorArg(box_id_dtype, "peer_lists"),
        VectorArg(coord_dtype, "ball_radii"),
        ] + [
        ScalarArg(coord_dtype, "bbox_min_"+ax)
        for ax in AXIS_NAMES[:dimensions]
        ] + [
        VectorArg(coord_dtype, "ball_"+ax)
        for ax in AXIS_NAMES[:dimensions]]

    from pyopencl.algorithm import ListOfListsBuilder
    area_query_knl = ListOfListsBuilder(
        actx.context,
        [("leaves", box_id_dtype)],
        str(template.render(**render_vars)),
        arg_decls=arg_decls,
        name_prefix="area_query",
        count_sharing={},
        complex_kernel=True)

    logger.debug("done building area query kernel")
    return area_query_knl


def build_area_query(
        actx: PyOpenCLArrayContext, tree: Tree,
        ball_centers, ball_radii, peer_lists=None) -> AreaQueryResult:
    r"""Given a set of :math:`l^\infty` "balls", this class helps build a
    look-up table from ball to leaf boxes that intersect with the ball.

    :arg ball_centers: an object array of coordinates. Their *dtype* must
        match *tree*'s :attr:`boxtree.Tree.coord_dtype`.
    :arg ball_radii: an array of positive numbers. Its *dtype* must match
        *tree*'s :attr:`boxtree.Tree.coord_dtype`.
    :arg peer_lists: may either be *None* or an instance of
        :class:`PeerListLookup` associated with `tree`.
    """

    # {{{ input check

    from pytools import single_valued
    if single_valued(bc.dtype for bc in ball_centers) != tree.coord_dtype:
        raise TypeError("ball_centers dtype must match tree.coord_dtype")

    if ball_radii.dtype != tree.coord_dtype:
        raise TypeError("ball_radii dtype must match tree.coord_dtype")

    from pytools import div_ceil
    # Avoid generating too many kernels.
    max_levels = div_ceil(tree.nlevels, 10) * 10

    if peer_lists is None:
        peer_lists = build_peer_list(actx, tree)

    if len(peer_lists.peer_list_starts) != tree.nboxes + 1:
        raise ValueError("size of peer lists must match with number of boxes")

    ball_id_dtype = tree.particle_id_dtype
    peer_list_idx_dtype = peer_lists.peer_list_starts.dtype

    # }}}

    # {{{ area query

    area_query_knl = get_area_query_kernel(
        actx,
        tree.dimensions, tree.coord_dtype, tree.box_id_dtype,
        ball_id_dtype, peer_list_idx_dtype, max_levels,
        tree.root_extent_stretch_factor)

    with ProcessLogger(logger, "area query"):
        result, _ = area_query_knl(
                actx.queue, len(ball_radii),
                tree.box_centers.data, tree.root_extent,
                tree.box_levels, tree.aligned_nboxes,
                tree.box_child_ids.data, tree.box_flags,
                peer_lists.peer_list_starts,
                peer_lists.peer_lists, ball_radii,
                *(tuple(tree.bounding_box[0]) + tuple(bc for bc in ball_centers)),
                allocator=actx.allocator,
                )

    # }}}

    result = AreaQueryResult(
            tree=tree,
            leaves_near_ball_starts=result["leaves"].starts,
            leaves_near_ball_lists=result["leaves"].lists)

    return actx.freeze(result)

# }}}


# {{{ area query transpose (leaves-to-balls) lookup build

class LeavesToBallsLookupBuilder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, actx: PyOpenCLArrayContext, tree: Tree,
                 ball_centers, ball_radii, peer_lists=None,
                 wait_for=None):
        from warnings import warn
        warn(f"'{type(self).__name__}' is deprecated and will be removed in 2023. "
            "Use 'build_leaves_to_balls_lookup' instead.",
            DeprecationWarning, stacklevel=2)

        return build_leaves_to_balls_lookup(
            actx, tree, ball_centers, ball_radii, peer_lists)


@dataclass_array_container
@dataclass(frozen=True)
class LeavesToBallsLookup:
    """
    .. attribute:: tree

        The :class:`boxtree.Tree` instance used to build this lookup.

    .. attribute:: balls_near_box_starts

        Indices into :attr:`balls_near_box_lists`.
        ``balls_near_box_lists[balls_near_box_starts[ibox]:
        balls_near_box_starts[ibox]+1]``
        results in a list of balls that overlap leaf box *ibox*.

        .. note:: Only leaf boxes have non-empty entries in this table. Nonetheless,
            this list is indexed by the global box index.

    .. attribute:: balls_near_box_lists
    """

    tree: Tree
    balls_near_box_starts: Array
    balls_near_box_lists: Array


def build_leaves_to_balls_lookup(
        actx: PyOpenCLArrayContext, tree: Tree,
        ball_centers, ball_radii, peer_lists=None) -> LeavesToBallsLookup:
    r"""Given a set of :math:`l^\infty` "balls", this builds a
    look-up table from leaf boxes to balls that overlap with each leaf box.

    :arg ball_centers: an object array of coordinates. Their *dtype* must
        match *tree*'s :attr:`boxtree.Tree.coord_dtype`.
    :arg ball_radii: an array of positive numbers. Its *dtype* must match
        *tree*'s :attr:`boxtree.Tree.coord_dtype`.
    :arg peer_lists: may either be *None* or an instance of
        :class:`PeerListLookup` associated with `tree`.
    """

    # {{{ check inputs

    from pytools import single_valued
    if single_valued(bc.dtype for bc in ball_centers) != tree.coord_dtype:
        raise TypeError("ball_centers dtype must match tree.coord_dtype")

    if ball_radii.dtype != tree.coord_dtype:
        raise TypeError("ball_radii dtype must match tree.coord_dtype")

    # }}}

    # {{{ build lookup

    from pytools import memoize_in

    @memoize_in(actx, (build_leaves_to_balls_lookup, tree.box_id_dtype))
    def get_starts_expander_kernel():
        return STARTS_EXPANDER_TEMPLATE.build(
                actx.context,
                type_aliases=(("idx_t", tree.box_id_dtype),))

    @memoize_in(actx, (build_leaves_to_balls_lookup, "key_value_sorter"))
    def get_key_value_sorter_kernel():
        from pyopencl.algorithm import KeyValueSorter
        return KeyValueSorter(actx.context)

    with ProcessLogger(logger, "leaves-to-balls lookup: run area query"):
        area_query = build_area_query(
            actx, tree, ball_centers, ball_radii, peer_lists)

        logger.debug("leaves-to-balls lookup: expand starts")

        nkeys = tree.nboxes
        nballs_p_1 = len(area_query.leaves_near_ball_starts)
        assert nballs_p_1 == len(ball_radii) + 1

        # We invert the area query in two steps:
        #
        # 1. Turn the area query result into (ball number, box number) pairs.
        #    This is done in the "starts expander kernel."
        #
        # 2. Key-value sort the (ball number, box number) pairs by box number.

        starts_expander_knl = get_starts_expander_kernel()
        expanded_starts = actx.empty(
                len(area_query.leaves_near_ball_lists), tree.box_id_dtype)
        evt = starts_expander_knl(
                expanded_starts,
                area_query.leaves_near_ball_starts,
                nballs_p_1,
                queue=actx.queue,
                )
        expanded_starts.add_event(evt)

        logger.debug("leaves-to-balls lookup: key-value sort")

        sorter_knl = get_key_value_sorter_kernel()
        balls_near_box_starts, balls_near_box_lists, _ = sorter_knl(
                actx.queue,
                # keys
                area_query.leaves_near_ball_lists,
                # values
                expanded_starts,
                nkeys, starts_dtype=tree.box_id_dtype,
                allocator=actx.allocator,
                )

    # }}}

    lookup = LeavesToBallsLookup(
            tree=tree,
            balls_near_box_starts=balls_near_box_starts,
            balls_near_box_lists=balls_near_box_lists)

    return actx.freeze(lookup)

# }}}


# {{{ space invader query build

class SpaceInvaderQueryBuilder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self,
            actx: PyOpenCLArrayContext, tree: Tree,
            ball_centers, ball_radii, peer_lists=None, wait_for=None):
        from warnings import warn
        warn(f"'{type(self).__name__}' is deprecated and will be removed in 2023. "
            "Use 'build_space_invader_query' instead.",
            DeprecationWarning, stacklevel=2)

        return build_space_invader_query(
            actx, tree, ball_centers, ball_radii, peer_lists)


@memoize_on_first_arg
def get_space_invader_query_kernel(
        actx: PyOpenCLArrayContext,
        dimensions: int,
        coord_dtype: "np.dtype",
        box_id_dtype: "np.dtype",
        peer_list_starts_dtype: "np.dtype",
        max_levels: int,
        root_extent_stretch_factor: float):
    return SPACE_INVADER_QUERY_TEMPLATE.generate(
            actx.context,
            dimensions,
            coord_dtype,
            box_id_dtype,
            peer_list_starts_dtype,
            max_levels,
            root_extent_stretch_factor=root_extent_stretch_factor)


def build_space_invader_query(
        actx: PyOpenCLArrayContext, tree: Tree,
        ball_centers, ball_radii, peer_lists=None) -> Array:
    r"""
    Given a set of :math:`l^\infty` "balls", this class helps build a look-up
    table which maps leaf boxes to the *outer space invader distance*.
    This is defined below but roughly, from the point of view
    of a leaf box, it is the farthest "leaf center to ball center" distance among
    all balls that intersect the leaf box.

    Formally, given a leaf box :math:`b`, the *outer space invader distance* is
    defined by the following expression (here :math:`d_\infty` is the
    :math:`\infty` norm):

    .. math::

        \max \left( \{ d_{\infty}(\text{center}(b), \text{center}(b^*))
        : b^* \text{ is a ball}, b^* \cap b \neq \varnothing \}
        \cup \{ 0 \} \right).

    :arg ball_centers: an object array of coordinates. Their *dtype* must
        match *tree*'s :attr:`boxtree.Tree.coord_dtype`.
    :arg ball_radii: an array of positive numbers. Its *dtype* must match
        *tree*'s :attr:`boxtree.Tree.coord_dtype`.
    :arg peer_lists: may either be *None* or an instance of
        :class:`PeerListLookup` associated with *tree*.

    :returns: an array with *dtype* same as the *tree*'s
        :attr:`boxtree.Tree.coord_dtype` and its shape is *(tree.nboxes,)*
        (see :attr:`boxtree.Tree.nboxes`). The entries of the array are
        indexed by the global box index and are as follows:

        * if *i* is not the index of a leaf box, *sqi[i] = 0*.
        * if *i* is the index of a leaf box, *sqi[i]* is the
            outer space invader distance for *i*.
    """
    # {{{ check inputs

    from pytools import single_valued

    if single_valued([bc.dtype for bc in ball_centers]) != tree.coord_dtype:
        raise TypeError("ball_centers dtype must match tree.coord_dtype")

    if ball_radii.dtype != tree.coord_dtype:
        raise TypeError("ball_radii dtype must match tree.coord_dtype")

    from pytools import div_ceil
    # Avoid generating too many kernels.
    max_levels = div_ceil(tree.nlevels, 10) * 10

    if peer_lists is None:
        peer_lists = build_peer_list(actx, tree)

    if len(peer_lists.peer_list_starts) != tree.nboxes + 1:
        raise ValueError("size of peer lists must match with number of boxes")

    # }}}

    # {{{ build query

    space_invader_query_knl = get_space_invader_query_kernel(
        actx,
        tree.dimensions, tree.coord_dtype, tree.box_id_dtype,
        peer_lists.peer_list_starts.dtype,
        max_levels, tree.root_extent_stretch_factor,
        )

    with ProcessLogger(logger, "space invader query"):
        outer_space_invader_dists = actx.zeros(tree.nboxes, np.float32)
        evt = space_invader_query_knl(
                *SPACE_INVADER_QUERY_TEMPLATE.unwrap_args(
                    tree, peer_lists,
                    ball_radii,
                    outer_space_invader_dists,
                    *tuple(bc for bc in ball_centers)),
                queue=actx.queue,
                range=slice(len(ball_radii)),
                )
        outer_space_invader_dists.add_event(evt)

        if tree.coord_dtype != np.dtype(np.float32):
            # The kernel output is always an array of float32 due to limited
            # support for atomic operations with float64 in OpenCL.
            # Here the output is cast to match the coord dtype.
            outer_space_invader_dists = (
                outer_space_invader_dists.astype(tree.coord_dtype))

    # }}}

    return actx.freeze(outer_space_invader_dists)

# }}}


# {{{ peer list build

class PeerListFinder:
    def __init__(self, *args, **kwargs):
        pass

    def __call__(self, actx: PyOpenCLArrayContext, tree: Tree, wait_for=None):
        from warnings import warn
        warn(f"'{type(self).__name__}' is deprecated and will be removed in 2023. "
            "Use 'build_peer_list' instead.",
            DeprecationWarning, stacklevel=2)

        return build_peer_list(actx, tree)


@dataclass_array_container
@dataclass(frozen=True)
class PeerListLookup:
    """
    .. attribute:: tree

        The :class:`boxtree.Tree` instance used to build this lookup.

    .. attribute:: peer_list_starts

        Indices into :attr:`peer_lists`.
        ``peer_lists[peer_list_starts[box_id]:peer_list_starts[box_id]+1]``
        contains the list of peer boxes of box `box_id`.

    .. attribute:: peer_lists

    .. versionadded:: 2016.1
    """

    tree: Tree
    peer_list_starts: Array
    peer_lists: Array


@memoize_on_first_arg
def get_peer_list_finder_kernel(
        actx: PyOpenCLArrayContext,
        dimensions: int,
        coord_dtype: "np.dtype",
        box_id_dtype: "np.dtype",
        max_levels: int):
    from pyopencl.tools import dtype_to_ctype

    from boxtree import box_flags_enum
    from boxtree.tools import AXIS_NAMES
    from boxtree.traversal import (
        TRAVERSAL_PREAMBLE_TEMPLATE, HELPER_FUNCTION_TEMPLATE)

    logger.debug("start building peer list finder kernel")

    template = Template(
        TRAVERSAL_PREAMBLE_TEMPLATE
        + HELPER_FUNCTION_TEMPLATE
        + PEER_LIST_FINDER_TEMPLATE,
        strict_undefined=True)

    render_vars = dict(
        np=np,
        dimensions=dimensions,
        dtype_to_ctype=dtype_to_ctype,
        box_id_dtype=box_id_dtype,
        particle_id_dtype=None,
        coord_dtype=coord_dtype,
        get_coord_vec_dtype=get_coord_vec_dtype,
        cvec_sub=partial(coord_vec_subscript_code, dimensions),
        max_levels=max_levels,
        AXIS_NAMES=AXIS_NAMES,
        box_flags_enum=box_flags_enum,
        debug=False,
        # For calls to the helper is_adjacent_or_overlapping()
        targets_have_extent=False,
        sources_have_extent=False)

    from boxtree.tools import VectorArg, ScalarArg
    arg_decls = [
        VectorArg(coord_dtype, "box_centers", with_offset=False),
        ScalarArg(coord_dtype, "root_extent"),
        VectorArg(np.uint8, "box_levels"),
        ScalarArg(box_id_dtype, "aligned_nboxes"),
        VectorArg(box_id_dtype, "box_child_ids", with_offset=False),
        VectorArg(box_flags_enum.dtype, "box_flags"),
    ]

    from pyopencl.algorithm import ListOfListsBuilder
    peer_list_finder_knl = ListOfListsBuilder(
        actx.context,
        [("peers", box_id_dtype)],
        str(template.render(**render_vars)),
        arg_decls=arg_decls,
        name_prefix="find_peer_lists",
        count_sharing={},
        complex_kernel=True)

    logger.debug("done building peer list finder kernel")
    return peer_list_finder_knl


def build_peer_list(actx: PyOpenCLArrayContext, tree: Tree) -> PeerListLookup:
    """Builds a look-up table from box numbers to peer boxes. The full definition
    [1]_ of a peer box is as follows:

    Given a box :math:`b_j` in a quad-tree, :math:`b_k` is a peer box of
    :math:`b_j` if it is

        1. adjacent to :math:`b_j`,

        2. of at least the same size as :math:`b_j` (i.e. at the same or a
        higher level than), and

        3. no child of :math:`b_k` satisfies the above two criteria.

    .. [1] Rachh, Manas, Andreas Klöckner, and Michael O'Neil. "Fast
       algorithms for Quadrature by Expansion I: Globally valid expansions."
    """

    from pytools import div_ceil

    # Round up level count--this gets included in the kernel as
    # a stack bound. Rounding avoids too many kernel versions.
    max_levels = div_ceil(tree.nlevels, 10) * 10

    peer_list_finder_knl = get_peer_list_finder_kernel(
        actx,
        tree.dimensions, tree.coord_dtype, tree.box_id_dtype,
        max_levels,
        )

    with ProcessLogger(logger, "find peer lists"):
        result, evt = peer_list_finder_knl(
                actx.queue, tree.nboxes,
                tree.box_centers.data, tree.root_extent,
                tree.box_levels, tree.aligned_nboxes,
                tree.box_child_ids.data, tree.box_flags,
                allocator=actx.allocator,
                )

    lookup = PeerListLookup(
            tree=tree,
            peer_list_starts=result["peers"].starts,
            peer_lists=result["peers"].lists)

    return actx.freeze(lookup)

# }}}

# vim: fdm=marker
