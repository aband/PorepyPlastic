"""Checkpointed plastic cantilever run with an elastic Newton predictor.

    OPENBLAS_NUM_THREADS=1 python -B -m prototype.cantilever_full_study

All equations, constitutive updates, Jacobians and acceptance checks are the
existing prototype implementation. Fixed Jacobian maps are reused. The initial
Newton guess is the proportional elastic increment, including retained faces;
this avoids starting the line-search merit at a very small applied-load residual
just before plastic activation. Patches are scoped to this single-process run.
An optional elastic prefix is solved once and sampled proportionally ONLY after
checking that every material point remained elastic. No plastic step is skipped.
The default linear backend is equilibrated SuperLU with physical-residual
refinement; --linear-solver native selects the shared solver for comparison.
Use --resume to continue from the latest accepted checkpoint in --output-dir.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import porepy as pp
import scipy.sparse as sps

import prototype.loading as loading
import prototype.newton as newton
from prototype.cantilever import CantileverResult, RESPONSE_COLUMNS, load_factors, make_case, mean_tip_displacement
from prototype.cantilever_plots import export_results
from prototype.elastic_bending import ElasticBendingCase, solve_elastic
from prototype.jacobian import RetainedAnalyticalJacobian
from prototype.state import RetainedState
from prototype.tpsa_state import TpsaState
from material.material_state import MaterialState
from material.tensor import stress, strain

FORCE_BALANCE_TOL = 2e-5  # N, checked before and after accumulating a step.


def boundary_force_imbalance(operators, residual):
    """Net external force from TOTAL residuals for this zero-body-force case.

    Summing cell momentum includes both sides of interior faces. Subtracting
    their face-balance residuals leaves precisely the external-boundary force.
    Neumann residual rows must not be subtracted: they omit applied traction.
    """
    nd = operators.dim
    cell = residual[operators.u_slice].reshape((nd, operators.num_cells), order='F')
    face = residual[operators.face_slice].reshape((nd, operators.num_free_faces), order='F')
    interior = ~np.isin(operators.free_faces, operators.boundary_faces)
    return float(np.linalg.norm(cell.sum(axis=1)-face[:, interior].sum(axis=1)))


def load_checkpoint(path: Path, case):
    """Restore all physical histories, validating geometry and state kinematics."""
    with np.load(path) as saved:
        if tuple(saved['shape']) != case.shape:
            raise ValueError('Checkpoint mesh shape differs from the requested case.')
        for name, axis in zip(('x_nodes', 'y_nodes', 'z_nodes'), case.axes, strict=True):
            np.testing.assert_allclose(saved[name], axis, rtol=0., atol=1e-13)
        sigma, epsilon_p, backstress = saved['stress'], saved['plastic_strain'], saved['backstress']
        history = [MaterialState(stress(sigma[:, :, cell]),
                                 strain(epsilon_p[:, :, cell]),
                                 stress(backstress[:, :, cell]), float(alpha))
                   for cell, alpha in enumerate(saved['alpha'])]
        physical = TpsaState(saved['x'], saved['bc_values'], saved['epsilon'], history,
                             saved['traction'], face_displacement=saved['face_displacement'])
        state = RetainedState(case.operators, physical, saved['side_traction'], saved['body_force'],
                              neumann_values=saved['neumann_values'])
        return state, saved['response'].copy()


def scaled_superlu(matrix, residual):
    """Equilibrated direct solve with physical-residual iterative refinement.

    Diagonal row/column scaling and a lower pivot threshold reduce fill for
    this slender beam. Neither the physical Jacobian nor Newton tolerances change.
    """
    physical = sps.csc_array(matrix)
    rows = 1./np.maximum(np.asarray(abs(physical).max(axis=1).toarray()).ravel(), np.finfo(float).tiny)
    equilibrated = sps.diags_array(rows) @ physical
    columns = 1./np.maximum(np.asarray(abs(equilibrated).max(axis=0).toarray()).ravel(), np.finfo(float).tiny)
    equilibrated = sps.csc_array(equilibrated @ sps.diags_array(columns))
    factor = sps.linalg.splu(equilibrated, permc_spec='COLAMD', diag_pivot_thresh=.1)
    delta = columns*factor.solve(-rows*residual)
    for _ in range(3):
        defect = physical@delta+residual
        if np.linalg.norm(defect) <= 1e-11*(1.+np.linalg.norm(residual)):
            break
        delta += columns*factor.solve(-rows*defect)
    if not np.all(np.isfinite(delta)):
        raise RuntimeError('Nonfinite SuperLU correction.')
    return delta


def save_checkpoint(folder: Path, case, state: RetainedState, response, result=None) -> None:
    """Save physical fields and complete constitutive history without pickling."""
    history = state.tpsa.material_states
    payload = dict(
        shape=np.asarray(case.shape), grading_ratio=case.grading_ratio,
        step=int(response[-1][0]), load_factor=float(response[-1][1]), response=np.stack(response),
        x=state.tpsa.x, bc_values=state.tpsa.bc_values, epsilon=state.tpsa.epsilon,
        face_displacement=state.tpsa.face_displacement, traction=state.tpsa.traction,
        side_traction=state.side_traction, body_force=state.body_force, neumann_values=state.neumann_values,
        stress=np.stack([h.stress.to_numpy(copy=False) for h in history], axis=2),
        plastic_strain=np.stack([h.plastic_strain.to_numpy(copy=False) for h in history], axis=2),
        backstress=np.stack([h.backstress.to_numpy(copy=False) for h in history], axis=2),
        alpha=np.asarray([h.alpha for h in history]),
        cell_centers=case.grid.cell_centers, cell_volumes=case.grid.cell_volumes,
        x_nodes=case.axes[0], y_nodes=case.axes[1], z_nodes=case.axes[2],
    )
    if result is not None:
        payload.update(residual_norms=result.residual_norms, step_lengths=result.step_lengths, thresholds=result.thresholds)
    temporary = folder/'prototype_checkpoint_latest.tmp.npz'
    np.savez_compressed(temporary, **payload)
    temporary.replace(folder/'prototype_checkpoint_latest.npz')
    factor = float(payload['load_factor'])
    if np.isclose(abs(factor), 1.) or (factor == 0. and payload['step'] > 0):
        label = 'A' if factor > 0 else 'B' if factor < 0 else 'final'
        np.savez_compressed(folder/f'prototype_checkpoint_{label}.npz', **payload)
    np.savetxt(folder/'prototype_response_progress.csv', payload['response'], delimiter=',',
               header=','.join(RESPONSE_COLUMNS), comments='')


def run(shape=(1500, 4, 8), *, steps=40, elastic_prefix=8, resume=False, linear_solver='superlu',
        output=Path('results/prototype_cantilever_3d/1500x4x8_40steps')) -> CantileverResult:
    factors = load_factors(steps)
    if not 0 <= elastic_prefix < steps//4:
        raise ValueError('elastic_prefix must precede the positive peak.')
    if linear_solver not in ('native', 'superlu'):
        raise ValueError('linear_solver must be native or superlu.')
    output.mkdir(parents=True, exist_ok=True)
    start = time.monotonic()
    status = dict(solver='prototype', shape=shape, grading_ratio=1., steps=steps,
                  traction_path_MPa=[0., .6, -.6, 0.], jacobian='analytical',
                  initial_guess='elastic predictor', elastic_prefix=elastic_prefix,
                  fixed_operator_cache=True, linear_solver=linear_solver, resumed=resume,
                  force_balance_tolerance_N=FORCE_BALANCE_TOL,
                  status='running', accepted_step=0)
    (output/'run_status.json').write_text(json.dumps(status, indent=2))
    case = make_case(*shape, grading_ratio=1.)
    ops = case.operators
    elastic_case = ElasticBendingCase('cantilever', ops,
        pp.BoundaryConditionVectorial(case.grid, ops.dirichlet_faces, 'dir'),
        case.boundary, .1*case.reference_neumann, case.axes, case.shape, traction=.06)
    print('Computing elastic predictor...', flush=True)
    elastic = solve_elastic(elastic_case)
    predictor = 10*ops.pack(elastic.x, elastic.face_displacement[:, ops.free_faces])
    if elastic_prefix and 10*factors[elastic_prefix-1]*elastic.metrics['max_vm_MPa'] >= 240.:
        raise ValueError('The requested elastic prefix reaches yield; use a shorter prefix.')
    print('Elastic tip at 0.06 MPa:', elastic.metrics['tip_mm'], 'mm', flush=True)
    fixed = {}
    class CachedJacobian(RetainedAnalyticalJacobian):
        def _assemble_fixed_maps(self):
            if not fixed:
                print('Building fixed retained operators...', flush=True)
                super()._assemble_fixed_maps()
                fixed.update({name: getattr(self, name) for name in ('B', 'T', 'P_side', 'L', 'P')})
            else:
                for name, matrix in fixed.items():
                    setattr(self, name, matrix)

    native_newton, native_linear = loading.solve_newton, newton._solve_linear_correction
    reference = case.reference_neumann
    reference_norm = np.sum(reference**2)
    stage = dict(step=0, factor=0.)
    def solve_with_predictor(*args, **kwargs):
        change = np.sum(kwargs['neumann_increment']*reference)/reference_norm
        kwargs['initial_z'] = change*predictor
        # The global sum can exceed its tolerance even when each block norm
        # passes on a fine mesh. Enforce it before the controller commits.
        kwargs['convergence_check'] = lambda evaluation: (
            boundary_force_imbalance(ops, evaluation.residual) <= FORCE_BALANCE_TOL
        )
        return native_newton(*args, **kwargs)

    blocks = (ops.u_slice, ops.r_slice, ops.p_slice, ops.face_slice)
    def logged_linear(matrix, residual, iteration):
        before = time.monotonic()
        norms = [float(np.linalg.norm(residual[b])) for b in blocks]
        print(f"Step {stage['step']}, correction {iteration+1}: residuals={norms}, nnz={matrix.nnz}", flush=True)
        with (output/'prototype_newton_iterations.jsonl').open('a') as stream:
            stream.write(json.dumps(dict(**stage, iteration=iteration, residual_norms=norms))+'\n')
        delta = (native_linear(matrix, residual, iteration) if linear_solver == 'native'
                 else scaled_superlu(matrix, residual))
        defect = matrix@delta+residual
        print('Linear solve seconds', time.monotonic()-before,
              'defect', [float(np.linalg.norm(defect[b])) for b in blocks], flush=True)
        return delta

    initial = RetainedState.zeros(ops)
    rows = [np.zeros(len(RESPONSE_COLUMNS))]
    start_step = elastic_prefix
    state_A = state_B = initial
    initial_factor = 0.
    if resume:
        initial, previous = load_checkpoint(output/'prototype_checkpoint_latest.npz', case)
        rows = list(previous)
        start_step, initial_factor = int(previous[-1, 0]), float(previous[-1, 1])
        if not 0 < start_step < steps:
            raise ValueError('Checkpoint must precede the final load step.')
        np.testing.assert_allclose(previous[1:, 1], factors[:start_step], rtol=0., atol=1e-14)
        state_A, _ = load_checkpoint(output/'prototype_checkpoint_A.npz', case) if start_step >= steps//4 else (initial, previous)
        if start_step >= 3*steps//4:
            state_B, _ = load_checkpoint(output/'prototype_checkpoint_B.npz', case)
        status.update(accepted_step=start_step, load_factor=initial_factor)
        print('Resuming after accepted nominal step', start_step, flush=True)
    controller = loading.LoadStepController(ops, case.material_points, initial, case.boundary,
        reference_neumann=reference, initial_load_factor=initial_factor,
        jacobian_factory=lambda old: CachedJacobian(ops, case.material_points, old),
        max_iterations=35, atol=(2e-6, 2e-7, 2e-7, 2e-6), rtol=(1e-9,)*4,
        line_search=newton.BacktrackingOptions(max_backtracks=16))
    boundary = np.isin(ops.side_faces, ops.boundary_faces)

    def row(step, factor, record):
        state = record.state
        alpha = np.asarray([p.alpha for p in state.tpsa.material_states])
        balance = float(np.linalg.norm(state.side_traction[:, boundary].sum(axis=1)))
        if balance > FORCE_BALANCE_TOL:
            raise RuntimeError(f'Boundary force balance failed: {balance:g} N.')
        return np.asarray([step, factor, .6*factor, mean_tip_displacement(case, state),
                           alpha.max(), np.mean(alpha>1e-10), record.newton.iterations, balance])

    loading.solve_newton, newton._solve_linear_correction = solve_with_predictor, logged_linear
    try:
        if elastic_prefix and not resume:
            factor = float(factors[elastic_prefix-1])
            stage.update(step=elastic_prefix, factor=factor)
            record = controller.advance(factor)
            final_elastic_row = row(elastic_prefix, factor, record)
            if any(h.alpha != 0. for h in record.state.tpsa.material_states):
                raise RuntimeError('Elastic-prefix initialization generated plastic history.')
            for step in range(1, elastic_prefix+1):
                ratio = factors[step-1]/factor
                entry = final_elastic_row.copy()
                entry[:4] = [step, factors[step-1], .6*factors[step-1], ratio*final_elastic_row[3]]
                entry[6] = 0 if step < elastic_prefix else record.newton.iterations
                entry[7] *= ratio
                rows.append(entry)
            save_checkpoint(output, case, record.state, rows, record.newton)
            status.update(accepted_step=elastic_prefix, load_factor=factor)
            (output/'run_status.json').write_text(json.dumps(status, indent=2))
            print('Elastic prefix accepted through', .6*factor, 'MPa; tip', final_elastic_row[3], flush=True)
        for step in range(start_step+1, steps+1):
            factor = float(factors[step-1])
            stage.update(step=step, factor=factor)
            record = controller.advance(factor)
            rows.append(row(step, factor, record))
            state = record.state
            save_checkpoint(output, case, state, rows, record.newton)
            if np.isclose(factor, 1.): state_A = state.copy()
            if np.isclose(factor, -1.): state_B = state.copy()
            status.update(accepted_step=step, load_factor=factor, elapsed_seconds=time.monotonic()-start)
            (output/'run_status.json').write_text(json.dumps(status, indent=2))
            print(f"ACCEPTED {step}/{steps}: traction={.6*factor:+.3f}, tip={rows[-1][3]:+.8f} mm, "
                  f"max alpha={rows[-1][4]:.6g}, plastic fraction={rows[-1][5]:.4%}, "
                  f"Newton={record.newton.iterations}, step lengths={record.newton.step_lengths}", flush=True)
        result = CantileverResult(case, np.stack(rows), state_A, state_B, controller.state, 'analytical')
        paths = export_results(result, [], output)
        metadata = json.loads(paths['summary'].read_text())
        metadata.update(solver='prototype', newton_initial_guess='proportional elastic predictor',
                        linear_solver=linear_solver, resumed=resume,
                        force_balance_tolerance_N=FORCE_BALANCE_TOL,
                        elastic_prefix_steps=elastic_prefix,
                        elastic_prefix_note='Prefix verified elastic; intermediate samples scaled from its converged endpoint. Zero Newton count means an elastic reused sample.')
        paths['summary'].write_text(json.dumps(metadata, indent=2))
        status['status'] = 'complete'
        return result
    except BaseException as error:
        status.update(status='failed', error=repr(error), pending_step=stage['step'], pending_factor=stage['factor'])
        raise
    finally:
        loading.solve_newton, newton._solve_linear_correction = native_newton, native_linear
        status['elapsed_seconds'] = time.monotonic()-start
        (output/'run_status.json').write_text(json.dumps(status, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--nx', type=int, default=1500)
    parser.add_argument('--ny', type=int, default=4)
    parser.add_argument('--nz', type=int, default=8)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--elastic-prefix', type=int, default=8)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--linear-solver', choices=('native', 'superlu'), default='superlu')
    parser.add_argument('--output-dir', type=Path, default=Path('results/prototype_cantilever_3d/1500x4x8_40steps'))
    args = parser.parse_args()
    run((args.nx,args.ny,args.nz), steps=args.steps, elastic_prefix=args.elastic_prefix,
        resume=args.resume, linear_solver=args.linear_solver, output=args.output_dir)


if __name__ == '__main__':
    main()
