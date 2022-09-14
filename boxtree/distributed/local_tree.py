__copyright__ = "Copyright (C) 2013 Andreas Kloeckner \
                 Copyright (C) 2018 Hao Gao"

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

import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

from pyopencl.tools import dtype_to_ctype
from pyopencl.elementwise import ElementwiseKernel

from arraycontext import Array, ArrayOrContainer
from pytools import memoize_on_first_arg
from mako.template import Template

from boxtree import Tree
from boxtree.array_context import PyOpenCLArrayContext, dataclass_array_container

import logging
logger = logging.getLogger(__name__)


# FIXME: The logic in this file has a lot in common with
# the particle filtering functionality that already exists.
# We should refactor this to make use of this commonality.
# https://documen.tician.de/boxtree/tree.html#filtering-the-lists-of-targets

# {{{ kernels

FETCH_LOCAL_PARTICLES_ARGUMENTS_TPL = Template("""
    __global const ${mask_t} *particle_mask,
    __global const ${mask_t} *particle_scan
    % for dim in range(ndims):
        , __global const ${coord_t} *particles_${dim}
    % endfor
    % for dim in range(ndims):
        , __global ${coord_t} *local_particles_${dim}
    % endfor
    % if particles_have_extent:
        , __global const ${coord_t} *particle_radii
        , __global ${coord_t} *local_particle_radii
    % endif
""", strict_undefined=True)

FETCH_LOCAL_PARTICLES_PRG_TPL = Template("""
    if(particle_mask[i]) {
        ${particle_id_t} des = particle_scan[i];
        % for dim in range(ndims):
            local_particles_${dim}[des] = particles_${dim}[i];
        % endfor
        % if particles_have_extent:
            local_particle_radii[des] = particle_radii[i];
        % endif
    }
""", strict_undefined=True)


@memoize_on_first_arg
def particle_mask_kernel(actx: PyOpenCLArrayContext, particle_id_dtype):
    return ElementwiseKernel(
        actx.context,
        arguments=Template("""
            __global char *responsible_boxes,
            __global ${particle_id_t} *box_particle_starts,
            __global ${particle_id_t} *box_particle_counts_nonchild,
            __global ${particle_id_t} *particle_mask
        """, strict_undefined=True).render(
            particle_id_t=dtype_to_ctype(particle_id_dtype)
        ),
        operation=Template("""
            if(responsible_boxes[i]) {
                for(${particle_id_t} pid = box_particle_starts[i];
                    pid < box_particle_starts[i]
                            + box_particle_counts_nonchild[i];
                    ++pid) {
                    particle_mask[pid] = 1;
                }
            }
        """).render(particle_id_t=dtype_to_ctype(particle_id_dtype))
        )


@memoize_on_first_arg
def mask_scan_kernel(actx: PyOpenCLArrayContext, particle_id_dtype):
    from pyopencl.scan import GenericScanKernel
    return GenericScanKernel(
        actx.context, particle_id_dtype,
        arguments=Template("""
            __global ${mask_t} *ary,
            __global ${mask_t} *scan
            """, strict_undefined=True).render(
            mask_t=dtype_to_ctype(particle_id_dtype)
        ),
        input_expr="ary[i]",
        scan_expr="a+b", neutral="0",
        output_statement="scan[i + 1] = item;"
        )


@memoize_on_first_arg
def fetch_local_particles_kernel(
        actx: PyOpenCLArrayContext,
        dimensions, particle_id_dtype, coord_dtype,
        particles_have_extent):
    return ElementwiseKernel(
        actx.context,
        FETCH_LOCAL_PARTICLES_ARGUMENTS_TPL.render(
            mask_t=dtype_to_ctype(particle_id_dtype),
            coord_t=dtype_to_ctype(coord_dtype),
            ndims=dimensions,
            particles_have_extent=particles_have_extent
        ),
        FETCH_LOCAL_PARTICLES_PRG_TPL.render(
            particle_id_t=dtype_to_ctype(particle_id_dtype),
            ndims=dimensions,
            particles_have_extent=particles_have_extent
        )
    )


@memoize_on_first_arg
def modify_target_flags_kernel(actx: PyOpenCLArrayContext, particle_id_dtype):
    from boxtree import box_flags_enum
    box_flag_t = dtype_to_ctype(box_flags_enum.dtype)

    return ElementwiseKernel(
        actx.context,
        Template("""
            __global ${particle_id_t} *box_target_counts_nonchild,
            __global ${particle_id_t} *box_target_counts_cumul,
            __global ${box_flag_t} *box_flags
        """).render(
            particle_id_t=dtype_to_ctype(particle_id_dtype),
            box_flag_t=box_flag_t
        ),
        Template(r"""
            // reset HAS_OWN_TARGETS and HAS_CHILD_TARGETS bits in the flag of
            // each box
            box_flags[i] &= (~${HAS_OWN_TARGETS});
            box_flags[i] &= (~${HAS_CHILD_TARGETS});

            // rebuild HAS_OWN_TARGETS and HAS_CHILD_TARGETS bits
            if(box_target_counts_nonchild[i]) box_flags[i] |= ${HAS_OWN_TARGETS};
            if(box_target_counts_nonchild[i] < box_target_counts_cumul[i])
                box_flags[i] |= ${HAS_CHILD_TARGETS};
        """).render(
            HAS_OWN_TARGETS=(
                "(" + box_flag_t + ") " + str(box_flags_enum.HAS_OWN_TARGETS)
            ),
            HAS_CHILD_TARGETS=(
                "(" + box_flag_t + ") " + str(box_flags_enum.HAS_CHILD_TARGETS)
            )
        )
    )


@dataclass(frozen=True)
class LocalParticlesAndLists:
    particles: ArrayOrContainer
    particle_radii: Optional[Array]
    box_particle_starts: Array
    box_particle_counts_nonchild: Array
    box_particle_counts_cumul: Array
    particle_idx: np.ndarray


def construct_local_particles_and_lists(
        actx: PyOpenCLArrayContext,
        dimensions, num_boxes, num_global_particles,
        particle_id_dtype, coord_dtype, particles_have_extent,
        box_mask,
        global_particles, global_particle_radii,
        box_particle_starts, box_particle_counts_nonchild,
        box_particle_counts_cumul):
    """This helper function generates particles (either sources or targets) of the
    local tree, and reconstructs list of lists indexing accordingly.
    """
    # {{{ calculate the particle mask

    particle_mask = actx.zeros(num_global_particles, dtype=particle_id_dtype)
    knl = particle_mask_kernel(actx, particle_id_dtype)
    knl(box_mask,
        box_particle_starts,
        box_particle_counts_nonchild,
        particle_mask,
        queue=actx.queue,
        allocator=actx.allocator,
        )

    # }}}

    # {{{ calculate the scan of the particle mask

    global_to_local_particle_index = actx.empty(
        num_global_particles + 1, dtype=particle_id_dtype)

    global_to_local_particle_index[0] = 0
    knl = mask_scan_kernel(actx, particle_id_dtype)
    knl(particle_mask, global_to_local_particle_index,
        queue=actx.queue,
        allocator=actx.allocator,
        )

    # }}}

    # {{{ fetch the local particles

    from pytools.obj_array import make_obj_array
    num_local_particles = actx.to_numpy(global_to_local_particle_index[-1]).item()
    local_particles = make_obj_array([
        actx.zeros(num_local_particles, coord_dtype)
        for _ in range(dimensions)
        ])

    from pytools.obj_array import make_obj_array
    local_particles = make_obj_array(local_particles)

    knl = fetch_local_particles_kernel(
        actx, dimensions, particle_id_dtype, coord_dtype,
        particles_have_extent=particles_have_extent,
        queue=actx.queue,
        allocator=actx.allocator,
        )

    if particles_have_extent:
        local_particle_radii = actx.empty(num_local_particles, dtype=coord_dtype)
        knl(
            particle_mask, global_to_local_particle_index,
            *global_particles.tolist(),
            *local_particles,
            global_particle_radii,
            local_particle_radii,
            queue=actx.queue,
            allocator=actx.allocator,
            )
    else:
        local_particle_radii = None
        knl(
            particle_mask, global_to_local_particle_index,
            *global_particles.tolist(),
            *local_particles,
            queue=actx.queue,
            allocator=actx.allocator,

    # {{{ construct the list of list indices

    local_box_particle_starts = global_to_local_particle_index[box_particle_starts]

    box_counts_all_zeros = actx.zeros(num_boxes, dtype=particle_id_dtype)

    local_box_particle_counts_nonchild = actx.np.where(
        box_mask, box_particle_counts_nonchild, box_counts_all_zeros)

    box_particle_ends_cumul = box_particle_starts + box_particle_counts_cumul

    local_box_particle_counts_cumul = (
        global_to_local_particle_index[box_particle_ends_cumul]
        - global_to_local_particle_index[box_particle_starts])

    # }}}

    particle_mask = actx.to_numpy(particle_mask).astype(bool)
    particle_idx = np.arange(num_global_particles)[particle_mask]

    return LocalParticlesAndLists(
        particles=local_particles,
        particle_radii=local_particle_radii,
        box_particle_starts=local_box_particle_starts,
        box_particle_counts_nonchild=local_box_particle_counts_nonchild,
        box_particle_counts_cumul=local_box_particle_counts_cumul,
        particle_idx=particle_idx)


@dataclass_array_container
@dataclass(frozen=True)
class LocalTree(Tree):
    """
    Inherits from :class:`boxtree.Tree`.

    .. attribute:: box_to_user_rank_starts

        ``box_id_t [nboxes + 1]``

    .. attribute:: box_to_user_rank_lists

        ``int32 [*]``

        A :ref:`csr` array, together with :attr:`box_to_user_rank_starts`.
        For each box, the list of ranks which own targets that *use* the
        multipole expansion at this box, via either List 3 or (possibly downward
        propagated from an ancestor) List 2.
    """

    box_to_user_rank_starts: Array
    box_to_user_rank_lists: Array

    responsible_boxes_list: Array
    responsible_boxes_mask: Array
    ancestor_mask: Array


def generate_local_tree(
        actx: PyOpenCLArrayContext,
        global_traversal, responsible_boxes_list, comm):
    """Generate the local tree for the current rank.

    This is an MPI-collective routine on *comm*.

    :arg global_traversal: Global :class:`boxtree.traversal.FMMTraversalInfo` object
        on host memory.
    :arg responsible_boxes_list: a :class:`numpy.ndarray` object containing the
        responsible boxes of the current rank.

    :return: a tuple of ``(local_tree, src_idx, tgt_idx)``, where ``local_tree`` is
        an object with class :class:`boxtree.distributed.local_tree.LocalTree` of the
        generated local tree, ``src_idx`` is the indices of the local sources in the
        global tree, and ``tgt_idx`` is the indices of the local targets in the
        global tree. ``src_idx`` and ``tgt_idx`` are needed for distributing source
        weights from root rank and assembling calculated potentials on the root rank.
    """
    global_tree = actx.thaw(global_traversal.tree)

    mpi_rank = comm.Get_rank()
    mpi_size = comm.Get_size()

    start_time = time.time()

    from boxtree.distributed.partition import get_box_masks
    box_masks = get_box_masks(actx, global_traversal, responsible_boxes_list)

    local_sources_and_lists = construct_local_particles_and_lists(
        actx, global_tree.dimensions, global_tree.nboxes,
        global_tree.nsources,
        global_tree.particle_id_dtype, global_tree.coord_dtype,
        global_tree.sources_have_extent,
        box_masks.point_src_boxes,
        global_tree.sources,
        global_tree.sources_radii if global_tree.sources_have_extent else None,
        global_tree.box_source_starts,
        global_tree.box_source_counts_nonchild,
        global_tree.box_source_counts_cumul)

    local_targets_and_lists = construct_local_particles_and_lists(
        actx, global_tree.dimensions, global_tree.nboxes,
        global_tree.ntargets,
        global_tree.particle_id_dtype, global_tree.coord_dtype,
        global_tree.targets_have_extent,
        box_masks.responsible_boxes,
        global_tree.targets,
        global_tree.target_radii if global_tree.targets_have_extent else None,
        global_tree.box_target_starts,
        global_tree.box_target_counts_nonchild,
        global_tree.box_target_counts_cumul)

    # {{{ compute the users of multipole expansions of each box on the root rank

    multipole_src_boxes_all_ranks = None
    if mpi_rank == 0:
        multipole_src_boxes_all_ranks = np.empty(
            (mpi_size, global_tree.nboxes),
            dtype=box_masks.multipole_src_boxes.dtype)
    comm.Gather(
        actx.to_numpy(box_masks.multipole_src_boxes),
        multipole_src_boxes_all_ranks, root=0)

    box_to_user_rank_starts = None
    box_to_user_rank_lists = None

    if mpi_rank == 0:
        multipole_src_boxes_all_ranks = actx.from_numpy(
            multipole_src_boxes_all_ranks)

        from boxtree.tools import mask_to_csr
        (box_to_user_rank_starts, box_to_user_rank_lists) = (
            mask_to_csr(
                actx, multipole_src_boxes_all_ranks.transpose(),
                list_dtype=np.int32))

        box_to_user_rank_starts = actx.to_numpy(box_to_user_rank_starts)
        box_to_user_rank_lists = actx.to_numpy(box_to_user_rank_lists)

        logger.debug("computing box_to_user: done")

    box_to_user_rank_starts = comm.bcast(box_to_user_rank_starts, root=0)
    box_to_user_rank_lists = comm.bcast(box_to_user_rank_lists, root=0)

    # }}}

    # {{{ Reconstruct the target box flags

    # Note: We do not change the source box flags despite the local tree may only
    # contain a subset of sources. This is because evaluating target potentials in
    # the responsible boxes of the current rank may depend on the multipole
    # expansions formed by souces in other ranks. Modifying the source box flags
    # could result in incomplete interaction lists.

    local_box_flags = actx.np.copy(global_tree.box_flags)
    knl = modify_target_flags_kernel(actx, global_tree.particle_id_dtype)
    knl(
        local_targets_and_lists.box_particle_counts_nonchild,
        local_targets_and_lists.box_particle_counts_cumul,
        local_box_flags,
        queue=actx.queue,
        allocator=actx.allocator,
        )

    # }}}

    local_tree = LocalTree(
        sources_are_targets=global_tree.sources_are_targets,
        sources_have_extent=global_tree.sources_have_extent,
        targets_have_extent=global_tree.targets_have_extent,

        particle_id_dtype=global_tree.particle_id_dtype,
        box_id_dtype=global_tree.box_id_dtype,
        coord_dtype=global_tree.coord_dtype,
        box_level_dtype=global_tree.box_level_dtype,

        root_extent=global_tree.root_extent,
        stick_out_factor=global_tree.stick_out_factor,
        extent_norm=global_tree.extent_norm,

        bounding_box=global_tree.bounding_box,
        level_start_box_nrs=global_tree.level_start_box_nrs,

        sources=local_sources_and_lists.particles,
        targets=local_targets_and_lists.particles,
        source_radii=(
                local_sources_and_lists.particle_radii
                if global_tree.sources_have_extent else None),
        target_radii=(
                local_targets_and_lists.particle_radii
                if global_tree.targets_have_extent else None),

        box_source_starts=(
            local_sources_and_lists.box_particle_starts),
        box_source_counts_nonchild=(
            local_sources_and_lists.box_particle_counts_nonchild),
        box_source_counts_cumul=(
            local_sources_and_lists.box_particle_counts_cumul),
        box_target_starts=(
            local_targets_and_lists.box_particle_starts),
        box_target_counts_nonchild=(
            local_targets_and_lists.box_particle_counts_nonchild),
        box_target_counts_cumul=(
            local_targets_and_lists.box_particle_counts_cumul),

        box_parent_ids=global_tree.box_parent_ids,
        box_child_ids=global_tree.box_child_ids,
        box_centers=global_tree.box_centers,
        box_levels=global_tree.box_levels,
        box_flags=local_box_flags,

        user_source_ids=None,
        sorted_target_ids=None,

        box_source_bounding_box_min=global_tree.box_source_bounding_box_min,
        box_source_bounding_box_max=global_tree.box_source_bounding_box_max,
        box_target_bounding_box_min=global_tree.box_target_bounding_box_min,
        box_target_bounding_box_max=global_tree.box_target_bounding_box_max,

        _is_pruned=global_tree._is_pruned,

        responsible_boxes_list=responsible_boxes_list,
        responsible_boxes_mask=box_masks.responsible_boxes,
        ancestor_mask=box_masks.ancestor_boxes,
        box_to_user_rank_starts=actx.from_numpy(box_to_user_rank_starts),
        box_to_user_rank_lists=actx.from_numpy(box_to_user_rank_lists),
    )

    logger.info("Generate local tree on rank {} in {} sec.".format(
        mpi_rank, str(time.time() - start_time)
    ))

    return (
        actx.freeze(local_tree),
        local_sources_and_lists.particle_idx,
        local_targets_and_lists.particle_idx)
