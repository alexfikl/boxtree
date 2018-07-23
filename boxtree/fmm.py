from __future__ import division

__copyright__ = "Copyright (C) 2012 Andreas Kloeckner"

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

import logging
logger = logging.getLogger(__name__)

from pytools import ProcessLogger, Record
import pyopencl as cl
import numpy as np
from collections import namedtuple


def drive_fmm(traversal, expansion_wrangler, src_weights, timing_data=None):
    """Top-level driver routine for a fast multipole calculation.

    In part, this is intended as a template for custom FMMs, in the sense that
    you may copy and paste its
    `source code <https://github.com/inducer/boxtree/blob/master/boxtree/fmm.py>`_
    as a starting point.

    Nonetheless, many common applications (such as point-to-point FMMs) can be
    covered by supplying the right *expansion_wrangler* to this routine.

    :arg traversal: A :class:`boxtree.traversal.FMMTraversalInfo` instance.
    :arg expansion_wrangler: An object exhibiting the
        :class:`ExpansionWranglerInterface`.
    :arg src_weights: Source 'density/weights/charges'.
        Passed unmodified to *expansion_wrangler*.
    :arg timing_data: Either *None*, or a :class:`dict` that is populated with
        timing information for the stages of the algorithm (in the form of
        :class:`TimingResult`), if such information is available.

    Returns the potentials computed by *expansion_wrangler*.

    """
    wrangler = expansion_wrangler

    # Interface guidelines: Attributes of the tree are assumed to be known
    # to the expansion wrangler and should not be passed.

    fmm_proc = ProcessLogger(logger, "qbx fmm")
    recorder = TimingRecorder()

    src_weights = wrangler.reorder_sources(src_weights)

    # {{{ "Step 2.1:" Construct local multipoles

    mpole_exps, timing_future = wrangler.form_multipoles(
            traversal.level_start_source_box_nrs,
            traversal.source_boxes,
            src_weights)

    recorder.add("form_multipoles", timing_future)

    # }}}

    # {{{ "Step 2.2:" Propagate multipoles upward

    mpole_exps, timing_future = wrangler.coarsen_multipoles(
            traversal.level_start_source_parent_box_nrs,
            traversal.source_parent_boxes,
            mpole_exps)

    recorder.add("coarsen_multipoles", timing_future)

    # mpole_exps is called Phi in [1]

    # }}}

    # {{{ "Stage 3:" Direct evaluation from neighbor source boxes ("list 1")

    potentials, timing_future = wrangler.eval_direct(
            traversal.target_boxes,
            traversal.neighbor_source_boxes_starts,
            traversal.neighbor_source_boxes_lists,
            src_weights)

    recorder.add("eval_direct", timing_future)

    # these potentials are called alpha in [1]

    # }}}

    # {{{ "Stage 4:" translate separated siblings' ("list 2") mpoles to local

    local_exps, timing_future = wrangler.multipole_to_local(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_siblings_starts,
            traversal.from_sep_siblings_lists,
            mpole_exps)

    recorder.add("multipole_to_local", timing_future)

    # local_exps represents both Gamma and Delta in [1]

    # }}}

    # {{{ "Stage 5:" evaluate sep. smaller mpoles ("list 3") at particles

    # (the point of aiming this stage at particles is specifically to keep its
    # contribution *out* of the downward-propagating local expansions)

    mpole_result, timing_future = wrangler.eval_multipoles(
            traversal.target_boxes_sep_smaller_by_source_level,
            traversal.from_sep_smaller_by_level,
            mpole_exps)

    recorder.add("eval_multipoles", timing_future)

    potentials = potentials + mpole_result

    # these potentials are called beta in [1]

    if traversal.from_sep_close_smaller_starts is not None:
        logger.debug("evaluate separated close smaller interactions directly "
                "('list 3 close')")

        direct_result, timing_future = wrangler.eval_direct(
                traversal.target_boxes,
                traversal.from_sep_close_smaller_starts,
                traversal.from_sep_close_smaller_lists,
                src_weights)

        recorder.add("eval_direct", timing_future)

        potentials = potentials + direct_result

    # }}}

    # {{{ "Stage 6:" form locals for separated bigger source boxes ("list 4")

    local_result, timing_future = wrangler.form_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            traversal.from_sep_bigger_starts,
            traversal.from_sep_bigger_lists,
            src_weights)

    recorder.add("form_locals", timing_future)

    local_exps = local_exps + local_result

    if traversal.from_sep_close_bigger_starts is not None:
        direct_result, timing_future = wrangler.eval_direct(
                traversal.target_or_target_parent_boxes,
                traversal.from_sep_close_bigger_starts,
                traversal.from_sep_close_bigger_lists,
                src_weights)

        recorder.add("eval_direct", timing_future)

        potentials = potentials + direct_result

    # }}}

    # {{{ "Stage 7:" propagate local_exps downward

    local_exps, timing_future = wrangler.refine_locals(
            traversal.level_start_target_or_target_parent_box_nrs,
            traversal.target_or_target_parent_boxes,
            local_exps)

    recorder.add("refine_locals", timing_future)

    # }}}

    # {{{ "Stage 8:" evaluate locals

    local_result, timing_future = wrangler.eval_locals(
            traversal.level_start_target_box_nrs,
            traversal.target_boxes,
            local_exps)

    recorder.add("eval_locals", timing_future)

    potentials = potentials + local_result

    # }}}

    result = wrangler.reorder_potentials(potentials)

    result = wrangler.finalize_potentials(result)

    fmm_proc.done()

    if timing_data is not None:
        timing_data.update(recorder.summarize())

    return result


# {{{ expansion wrangler interface

class ExpansionWranglerInterface:
    """Abstract expansion handling interface for use with :func:`drive_fmm`.

    See this
    `test code <https://github.com/inducer/boxtree/blob/master/test/test_fmm.py>`_
    for a very simple sample implementation.

    Will usually hold a reference (and thereby be specific to) a
    :class:`boxtree.Tree` instance.

    Functions that support returning timing data return a value supporting the
    :class:`TimingFuture` interface.

    .. versionchanged:: 2018.1

        Changed (a subset of) functions to return timing data.
    """

    def multipole_expansion_zeros(self):
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        """

    def local_expansion_zeros(self):
        """Return an expansions array (which must support addition)
        capable of holding one multipole or local expansion for every
        box in the tree.
        """

    def output_zeros(self):
        """Return a potentials array (which must support addition) capable of
        holding a potential value for each target in the tree. Note that
        :func:`drive_fmm` makes no assumptions about *potential* other than
        that it supports addition--it may consist of potentials, gradients of
        the potential, or arbitrary other per-target output data.
        """

    def reorder_sources(self, source_array):
        """Return a copy of *source_array* in
        :ref:`tree source order <particle-orderings>`.
        *source_array* is in user source order.
        """

    def reorder_potentials(self, potentials):
        """Return a copy of *potentials* in
        :ref:`user target order <particle-orderings>`.
        *source_weights* is in tree target order.
        """

    def form_multipoles(self, level_start_source_box_nrs, source_boxes, src_weights):
        """Return an expansions array (compatible with
        :meth:`multipole_expansion_zeros`)
        containing multipole expansions in *source_boxes* due to sources
        with *src_weights*.
        All other expansions must be zero.

        :return: A pair (*mpoles*, *timing_future*).
        """

    def coarsen_multipoles(self, level_start_source_parent_box_nrs,
            source_parent_boxes, mpoles):
        """For each box in *source_parent_boxes*,
        gather (and translate) the box's children's multipole expansions in
        *mpole* and add the resulting expansion into the box's multipole
        expansion in *mpole*.

        :returns: A pair (*mpoles*, *timing_future*).
        """

    def eval_direct(self, target_boxes, neighbor_sources_starts,
            neighbor_sources_lists, src_weights):
        """For each box in *target_boxes*, evaluate the influence of the
        neighbor sources due to *src_weights*, which use :ref:`csr` and are
        indexed like *target_boxes*.

        :returns: A pair (*pot*, *timing_future*), where *pot* is a
            a new potential array, see :meth:`output_zeros`.
        """

    def multipole_to_local(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes,
            starts, lists, mpole_exps):
        """For each box in *target_or_target_parent_boxes*, translate and add
        the influence of the multipole expansion in *mpole_exps* into a new
        array of local expansions.  *starts* and *lists* use :ref:`csr`, and
        *starts* is indexed like *target_or_target_parent_boxes*.

        :returns: A pair (*pot*, *timing_future*) where *pot* is
            a new (local) expansion array, see :meth:`local_expansion_zeros`.
        """

    def eval_multipoles(self,
            target_boxes_by_source_level, from_sep_smaller_by_level, mpole_exps):
        """For a level *i*, each box in *target_boxes_by_source_level[i]*, evaluate
        the multipole expansion in *mpole_exps* in the nearby boxes given in
        *from_sep_smaller_by_level*, and return a new potential array.
        *starts* and *lists* in *from_sep_smaller_by_level[i]* use :ref:`csr`
        and *starts* is indexed like *target_boxes_by_source_level[i]*.

        :returns: A pair (*pot*, *timing_future*) where *pot* is a new potential
            array, see :meth:`output_zeros`.
        """

    def form_locals(self,
            level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, starts, lists, src_weights):
        """For each box in *target_or_target_parent_boxes*, form local
        expansions due to the sources in the nearby boxes given in *starts* and
        *lists*, and return a new local expansion array.  *starts* and *lists*
        use :ref:`csr` and *starts* is indexed like
        *target_or_target_parent_boxes*.

        :returns: A pair (*pot*, *timing_future*) where *pot* is a new
            local expansion array, see :meth:`local_expansion_zeros`.
        """

    def refine_locals(self, level_start_target_or_target_parent_box_nrs,
            target_or_target_parent_boxes, local_exps):
        """For each box in *child_boxes*,
        translate the box's parent's local expansion in *local_exps* and add
        the resulting expansion into the box's local expansion in *local_exps*.

        :returns: A pair (*local_exps*, *timing_future*).
        """

    def eval_locals(self, level_start_target_box_nrs, target_boxes, local_exps):
        """For each box in *target_boxes*, evaluate the local expansion in
        *local_exps* and return a new potential array.

        :returns: A pair (*pot*, *timing_future*) where *pot* is a new potential
            array, see :meth:`output_zeros`.
        """

    def finalize_potentials(self, potentials):
        """
        Postprocess the reordered potentials. This is where global scaling
        factors could be applied. This is distinct from :meth:`reorder_potentials`
        because some derived FMMs (notably the QBX FMM) do their own reordering.
        """

# }}}


# {{{ timing result

class TimingResult(Record):
    """
    .. attribute:: wall_elapsed
    .. attribute:: process_elapsed
    """

    def __init__(self, wall_elapsed, process_elapsed):
        Record.__init__(self,
                wall_elapsed=wall_elapsed,
                process_elapsed=process_elapsed)

# }}}


# {{{ timing future

class TimingFuture(object):
    """Returns timing data for a potentially asynchronous operation.

    .. automethod:: result
    .. automethod:: done
    """

    def result(self):
        """Return a :class:`TimingResult`. May block."""
        raise NotImplementedError

    def done(self):
        """Return *True* if the operation is complete."""
        raise NotImplementedError

# }}}


# {{{ timing recorder

class TimingRecorder(object):

    def __init__(self):
        from collections import defaultdict
        self.futures = defaultdict(list)

    def add(self, description, future):
        self.futures[description].append(future)

    def merge(self, result1, result2):
        wall_elapsed = None
        process_elapsed = None

        if None not in (result1.wall_elapsed, result2.wall_elapsed):
            wall_elapsed = result1.wall_elapsed + result2.wall_elapsed
        if None not in (result1.process_elapsed, result2.process_elapsed):
            process_elapsed = result1.process_elapsed + result2.process_elapsed

        return TimingResult(wall_elapsed, process_elapsed)

    def summarize(self):
        result = {}

        for description, futures_list in self.futures.items():
            futures = iter(futures_list)

            timing_result = next(futures).result()
            for future in futures:
                timing_result = self.merge(timing_result, future.result())

            result[description] = timing_result

        return result

# }}}


FMMParameters = namedtuple(
    "FMMParameters",
    ['ncoeffs_fmm_by_level',
     'translation_source_power',
     'translation_target_power',
     'translation_max_power']
)


class PerformanceCounter:

    def __init__(self, traversal, wrangler, uses_pde_expansions):
        self.traversal = traversal
        self.wrangler = wrangler
        self.uses_pde_expansions = uses_pde_expansions

        self.parameters = self.get_fmm_parameters(
            traversal.tree.dimensions,
            uses_pde_expansions,
            wrangler.level_nterms
        )

    @staticmethod
    def xlat_cost(p_source, p_target, parameters):
        """
        :param p_source: A numpy array of numbers of source terms
        :return: The same shape as *p_source*
        """
        return (
                p_source ** parameters.translation_source_power
                * p_target ** parameters.translation_target_power
                * np.maximum(p_source, p_target) ** parameters.translation_max_power
        )

    @staticmethod
    def get_fmm_parameters(dimensions, use_pde_expansions, level_nterms):
        if use_pde_expansions:
            ncoeffs_fmm_by_level = level_nterms ** (dimensions - 1)

            if dimensions == 2:
                translation_source_power = 1
                translation_target_power = 1
                translation_max_power = 0
            elif dimensions == 3:
                # Based on a reading of FMMlib, i.e. a point-and-shoot FMM.
                translation_source_power = 0
                translation_target_power = 0
                translation_max_power = 3
            else:
                raise ValueError("Don't know how to estimate expansion complexities "
                                 "for dimension %d" % dimensions)

        else:
            ncoeffs_fmm_by_level = level_nterms ** dimensions

            translation_source_power = dimensions
            translation_target_power = dimensions
            translation_max_power = 0

        return FMMParameters(
            ncoeffs_fmm_by_level=ncoeffs_fmm_by_level,
            translation_source_power=translation_source_power,
            translation_target_power=translation_target_power,
            translation_max_power=translation_max_power
        )

    def count_nsources_by_level(self):
        """
        :return: A numpy array of share (tree.nlevels,) such that the ith index
            documents the number of sources on level i.
        """
        tree = self.traversal.tree

        nsources_by_level = np.empty((tree.nlevels,), dtype=np.int32)

        for ilevel in range(tree.nlevels):
            start_ibox = tree.level_start_box_nrs[ilevel]
            end_ibox = tree.level_start_box_nrs[ilevel + 1]
            count = 0

            for ibox in range(start_ibox, end_ibox):
                count += tree.box_source_counts_nonchild[ibox]

            nsources_by_level[ilevel] = count

        return nsources_by_level

    def count_nters_fmm_total(self):
        """
        :return: total number of terms formed across all levels during form_multipole
        """
        nsources_by_level = self.count_nsources_by_level()

        ncoeffs_fmm_by_level = self.parameters.ncoeffs_fmm_by_level

        nterms_fmm_total = np.sum(nsources_by_level * ncoeffs_fmm_by_level)

        return nterms_fmm_total

    def count_direct(self, use_global_idx=False):
        """
        :return: If *use_global_idx* is True, return a numpy array of shape
            (tree.nboxes,) such that the ith entry represents the workload from
            direct evaluation on box i. If *use_global_idx* is False, return a numpy
            array of shape (ntarget_boxes,) such that the ith entry represents the
            workload on *target_boxes* i.
        """
        traversal = self.traversal
        tree = traversal.tree

        if use_global_idx:
            direct_workload = np.zeros((tree.nboxes,), dtype=np.int64)
        else:
            ntarget_boxes = len(traversal.target_boxes)
            direct_workload = np.zeros((ntarget_boxes,), dtype=np.int64)

        for itgt_box, tgt_ibox in enumerate(traversal.target_boxes):
            ntargets = tree.box_target_counts_nonchild[tgt_ibox]
            nsources = 0

            start, end = traversal.neighbor_source_boxes_starts[itgt_box:itgt_box+2]

            for src_ibox in traversal.neighbor_source_boxes_lists[start:end]:
                nsources += tree.box_source_counts_nonchild[src_ibox]

            if traversal.from_sep_close_smaller_starts is not None:
                start, end = (
                    traversal.from_sep_close_smaller_starts[itgt_box:itgt_box+2])

                for src_ibox in traversal.from_sep_close_smaller_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            if traversal.from_sep_close_bigger_starts is not None:
                start, end = (
                    traversal.from_sep_close_bigger_starts[itgt_box:itgt_box+2])

                for src_ibox in traversal.from_sep_close_bigger_lists[start:end]:
                    nsources += tree.box_source_counts_nonchild[src_ibox]

            count = nsources * ntargets

            if use_global_idx:
                direct_workload[tgt_ibox] = count
            else:
                direct_workload[itgt_box] = count

        return direct_workload

    def count_m2l(self, use_global_idx=False):
        """
        :return: If *use_global_idx* is True, return a numpy array of shape
            (tree.nboxes,) such that the ith entry represents the workload from
            multipole to local expansion on box i. If *use_global_idx* is False,
            return a numpy array of shape (ntarget_or_target_parent_boxes,) such that
            the ith entry represents the workload on *target_or_target_parent_boxes*
            i.
        """
        trav = self.traversal
        wrangler = self.wrangler
        parameters = self.parameters

        ntarget_or_target_parent_boxes = len(trav.target_or_target_parent_boxes)

        if use_global_idx:
            nm2l = np.zeros((trav.tree.nboxes,), dtype=np.intp)
        else:
            nm2l = np.zeros((ntarget_or_target_parent_boxes,), dtype=np.intp)

        for itgt_box, tgt_ibox in enumerate(trav.target_or_target_parent_boxes):
            start, end = trav.from_sep_siblings_starts[itgt_box:itgt_box+2]
            from_sep_siblings_level = trav.tree.box_levels[
                trav.from_sep_siblings_lists[start:end]
            ]

            if start == end:
                continue

            tgt_box_level = trav.tree.box_levels[tgt_ibox]

            from_sep_siblings_nterms = wrangler.level_nterms[from_sep_siblings_level]
            tgt_box_nterms = wrangler.level_nterms[tgt_box_level]

            from_sep_siblings_costs = self.xlat_cost(
                from_sep_siblings_nterms, tgt_box_nterms, parameters)

            if use_global_idx:
                nm2l[tgt_ibox] += np.sum(from_sep_siblings_costs)
            else:
                nm2l[itgt_box] += np.sum(from_sep_siblings_costs)

        return nm2l


class PerformanceModel:

    def __init__(self, cl_context, wrangler_factory, uses_pde_expansions):
        self.cl_context = cl_context
        self.wrangler_factory = wrangler_factory
        self.uses_pde_expansions = uses_pde_expansions

        self.time_result = []

        from pyopencl.clrandom import PhiloxGenerator
        self.rng = PhiloxGenerator(cl_context)

    def time_performance(self, traversal):
        wrangler = self.wrangler_factory(traversal.tree)

        counter = PerformanceCounter(traversal, wrangler, self.uses_pde_expansions)

        # Record useful metadata for assembling performance data
        timing_data = {
            "nterms_fmm_total": counter.count_nters_fmm_total(),
            "direct_workload": np.sum(counter.count_direct()),
            "direct_nsource_boxes": traversal.neighbor_source_boxes_starts[-1],
            "m2l_workload": np.sum(counter.count_m2l())
        }

        # Generate random source weights
        with cl.CommandQueue(self.cl_context) as queue:
            source_weights = self.rng.uniform(
                queue,
                traversal.tree.nsources,
                traversal.tree.coord_dtype
            ).get()

        # Time a FMM run
        drive_fmm(traversal, wrangler, source_weights, timing_data=timing_data)

        self.time_result.append(timing_data)

    def form_multipoles_model(self, wall_time=True):
        return self.linear_regression("form_multipoles", ["nterms_fmm_total"],
                                       wall_time=wall_time)

    def eval_direct_model(self, wall_time=True):
        return self.linear_regression(
            "eval_direct",
            ["direct_workload", "direct_nsource_boxes"],
            wall_time=wall_time)

    def linear_regression(self, y_name, x_name, wall_time=True):
        """
            :arg y_name: Name of the depedent variable
            :arg x_name: A list of names of independent variables
        """
        nresult = len(self.time_result)
        nvariables = len(x_name)

        if nresult < 1:
            raise RuntimeError("Please run FMM at lease once using time_performance"
                               "before forming models.")
        elif nresult == 1:
            result = self.time_result[0]

            if wall_time:
                dependent_value = result[y_name].wall_elapsed
            else:
                dependent_value = result[y_name].process_elapsed

            independent_value = result[x_name[0]]
            coeff = dependent_value / independent_value

            return (coeff,) + tuple(0.0 for _ in range(nvariables - 1))
        else:
            dependent_value = np.empty((nresult,), dtype=float)
            coeff_matrix = np.empty((nresult, nvariables + 1), dtype=float)

            for iresult, result in enumerate(self.time_result):
                if wall_time:
                    dependent_value[iresult] = result[y_name].wall_elapsed
                else:
                    dependent_value[iresult] = result[y_name].process_elapsed

                for icol, variable_name in enumerate(x_name):
                    coeff_matrix[iresult, icol] = result[variable_name]

            coeff_matrix[:, -1] = 1

            from numpy.linalg import lstsq
            coeff = lstsq(coeff_matrix, dependent_value, rcond=-1)[0]

            return coeff


# vim: filetype=pyopencl:fdm=marker
