from dataclasses import dataclass

import porepy as pp
import numpy as np

from core.layout import FloatArray, DofLayout
from core.operators import RetainedFaceOperators
from core.tpsa_wrapper import ElasticTpsa


@dataclass(frozen=True, eq=False)
class LoadIncrement:
    """Loading changes from the last equilibrated committed state."""

    # Shape: (dim, number of Dirichlet faces).
    # Ordered according to layout.dirichlet_faces.
    dirichlet_increment: FloatArray

    # Shape: (dim, number of Neumann faces).
    # Outward INTEGRATED traction increments, in neumann_faces order.
    neumann_increment: FloatArray

    # Shape: (layout.cell_size,).
    # Integrated sources ordered as [b_u, b_r, b_p].
    # Normally the rotation and pressure source blocks are zero.
    cell_source_increment: FloatArray

@dataclass(eq=False)
class ResidualEvaluation:
    """Matching outputs from one candidate evaluation."""

    residual: FloatArray           # (layout.size,)
    face_increment: FloatArray     # (dim, num_faces)
    traction_increment: FloatArray # (dim, num_sides)

class ResidualAssembler:
    """Assemble retained cell/face balances with fixed elastic operators."""

    def __init__(
        self,
        operators: RetainedFaceOperators,
        elastic: ElasticTpsa,
    ) -> None:
        self.operators = operators
        self.elastic = elastic
        self.layout = operators.layout

        self._validate_setup()

        # Reuse TPSA's existing V/mu and V/lambda blocks.
        self.accumulation = elastic.matrices.accum

        boundary = operators.grid.get_all_boundary_faces()
        self.neumann_faces = np.setdiff1d(
            boundary, self.layout.dirichlet_faces
        )

        # Locations of Neumann faces within the free-face residual block.
        self.neumann_positions = np.searchsorted(
            self.layout.free_faces, self.neumann_faces
        )

    def evaluate_elastic(
        self,
        z: FloatArray,
        loads: LoadIncrement,
    ) -> ResidualEvaluation:
        """Evaluate elastic fields and assemble their residual."""
        self._validate_inputs(z, loads)
        values = np.asarray(z, dtype=np.float64)

        face_increment = self.operators.face_displacement_increment(
            values, loads.dirichlet_increment
        )
        traction_increment = self.operators.elastic_one_sided_traction(
            values, loads.dirichlet_increment
        )

        residual = self.assemble(
            values, face_increment, traction_increment, loads
        )

        return ResidualEvaluation(
            residual=residual,
            face_increment=face_increment,
            traction_increment=traction_increment,
        )

    def assemble(
        self,
        z: FloatArray,
        face_increment: FloatArray,
        traction_increment: FloatArray,
        loads: LoadIncrement,
    ) -> FloatArray:
        """Assemble balances from validated candidate fields."""
        cell_flux = self._cell_flux_balance(
            face_increment, traction_increment
        )

        cell_residual = (
            cell_flux
            - self.accumulation @ z[self.layout.cell_slice]
            - loads.cell_source_increment
        )

        face_residual = self._free_face_balance(
            traction_increment, loads.neumann_increment
        )

        residual = self.layout.pack(cell_residual, face_residual)

        if not np.all(np.isfinite(residual)):
            raise ValueError("Residual must contain finite values.")

        return residual

    def _cell_flux_balance(
        self,
        face_increment: FloatArray,
        traction_increment: FloatArray,
    ) -> FloatArray:
        """Return cell flux sums."""

        operators = self.operators
        layout    = self.layout
        nd, nr, nc = layout.dim, layout.rotation_dim, layout.num_cells

        if (
            np.iscomplexobj(face_increment)
            or np.iscomplexobj(traction_increment)
        ):
            raise ValueError("Face increments and tractions must be real.")

        displacement = np.asarray(face_increment, dtype=np.float64)
        traction = np.asarray(traction_increment, dtype=np.float64)

        for name, values, shape in (
            ("face_increment", displacement, (nd, layout.num_faces)),
            ("traction_increment", traction, (nd, operators.num_sides)),
        ):
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} must be finite with shape {shape}."
                )

        cells = operators.side_cells
        normals = operators.side_normals
        areas = operators.side_areas
        side_displacement = displacement[:,operators.side_faces]

        result = np.empty(layout.cell_size, dtype=np.float64)

        # Views of components
        momentum = result[layout.u_slice].reshape((nd,nc), order="F")
        rotation = result[layout.r_slice].reshape((nr,nc), order="F")

        with np.errstate(over="ignore",invalid="ignore"):
            # Momentum flux
            for component in range(nd):
                momentum[component] = np.bincount(
                    cells,
                    weights=traction[component],
                    minlength=nc,
                )

            # Rotation flux
            if nd==2:
                rotation_flux = (
                    areas * (
                        side_displacement[0] * normals[1]
                        - side_displacement[1] * normals[0]
                    )
                )[None, :]
            else:
                rotation_flux = areas[None, :] * np.cross(
                    side_displacement, normals, axis=0
                )

            for component in range(nr):
                rotation[component] = np.bincount(
                    cells,
                    weights=rotation_flux[component],
                    minlength=nc
                )

            # Normal displacement flux
            pressure_flux = areas * np.sum(
                side_displacement * normals, axis=0
            )
            result[layout.p_slice] = np.bincount(
                cells,
                weights=pressure_flux,
                minlength=nc,
            )

        if not np.all(np.isfinite(result)):
            raise ValueError("Cell flux sums must remain finite.")

        return result

    def _free_face_balance(
        self,
        traction_increment: FloatArray,
        neumann_increment: FloatArray,
    ) -> FloatArray:
        """Return (dim, num_free_faces) traction-balance residuals."""
        operators = self.operators
        layout = self.layout
        nd, nf = layout.dim, layout.num_faces

        if (
            np.iscomplexobj(traction_increment)
            or np.iscomplexobj(neumann_increment)
        ):
            raise ValueError("Traction increments must be real.")

        traction = np.asarray(traction_increment, dtype=np.float64)
        neumann  = np.asarray(neumann_increment, dtype=np.float64)

        for name, values, shape in (
            ("traction_increment", traction, (nd, operators.num_sides)),
            ("neumann_increment", neumann, (nd, self.neumann_faces.size)),
        ):
            if values.shape != shape or not np.all(np.isfinite(values)):
                raise ValueError(
                    f"{name} must be finite with shape {shape}."
                )

        result = np.empty(
            (nd, layout.free_faces.size), dtype=np.float64
        )

        with np.errstate(over="ignore", invalid="ignore"):
            # Interior faces receive two contributions
            # Boundary faces receive one contribution
            for component in range(nd):
                face_sum = np.bincount(
                    operators.side_faces,
                    weights=traction[component],
                    minlength=nf,
                )
                result[component] = face_sum[layout.free_faces]

            result[:, self.neumann_positions] -= neumann

        if not np.all(np.isfinite(result)):
            raise ValueError("Free-face residuals must remian finite.")

        return result

    def _validate_setup(self) -> None:
        """Check shared grid, boundary partition, mu, and initialized coefficients."""
        operators = self.operators
        elastic = self.elastic
        layout = self.layout

        if operators.grid is not elastic.grid:
            raise ValueError(
                "Retained operators and TPSA must be defined on the same grid."
            )

        # Traction evaluateion requires initialized elastic coefficients
        if (
            operators.mu is None
            or operators.side_stiffness is None
            or operators.rotation_transfer is None
        ):
            raise ValueError(
                "Initialize retained operators with mu before"
                "Constructing the residual assembler."
            )

        # Reuse the existing boundary-condition and layout validation.
        expected = DofLayout.from_grid(elastic.grid, elastic.bc)

        for name in (
            "dim",
            "rotation_dim",
            "num_cells",
            "num_faces",
            "cell_size",
            "size",
            "u_slice",
            "r_slice",
            "p_slice",
            "face_slice",
        ):
            if getattr(layout, name) != getattr(expected, name):
                raise ValueError(f"Layout {name} is incompatiable with TPSA counterpart.")

        for name in (
            "dirichlet_faces",
            "free_faces",
            "face_dofs",
        ):
            if not np.array_equal(
                getattr(layout, name), getattr(expected, name)
            ): 
                raise ValueError(
                    f"Layout {name} does not match TPSA boundary numbering."
                )

        # Traction and accumulation must use identical reference elasticity.
        parameters = elastic.data[pp.PARAMETERS][elastic.keyword]
        tensor = parameters["fourth_order_tensor"]

        if not isinstance(tensor, pp.FourthOrderTensor):
            raise ValueError(
                "My tensor does not match Porepy FourthOrderTensor."
                )

        if not np.array_equal(operators.mu, tensor.mu):
            raise ValueError("Cellwise mu does not match.")

        accumulation = elastic.matrices.accum
        expected_shape = (layout.cell_size, layout.cell_size)

        if accumulation.shape != expected_shape:
            raise ValueError(
                f"Accumulation matrix must have shape {expected_shape}."
            )

        if (
            np.iscomplexobj(accumulation.data)
            or not np.all(np.isfinite(accumulation.data))
        ):
            raise ValueError(
                "Accumulation matrix must contain real, finite values."
            )

    def _validate_inputs(
        self,
        z: FloatArray,
        loads: LoadIncrement,
    ) -> None:
        """Validate candidate increments and loads without modifying them."""
        layout = self.layout

        checks = (
            ("z", z, (layout.size,)),
            (
                "dirichlet_increment",
                loads.dirichlet_increment,
                (layout.dim, layout.dirichlet_faces.size),
            ),
            (
                "neumann_increment",
                loads.neumann_increment,
                (layout.dim, self.neumann_faces.size),
            ),
            (
                "cell_source_increment",
                loads.cell_source_increment,
                (layout.cell_size,),
            ),
        )

        for name, values, expected_shape in checks:
            array = np.asarray(values)

            # Accept real floating-point and signed/unsigned integer data.
            # Reject complex, string, object, and Boolean arrays.
            if array.dtype.kind not in ("f", "i", "u"):
                raise ValueError(
                    f"{name} must contain real numeric values."
                )

            if array.shape != expected_shape:
                raise ValueError(
                    f"{name} must have shape {expected_shape}; "
                    f"got {array.shape}."
                )

            if not np.all(np.isfinite(array)):
                raise ValueError(
                    f"{name} must contain only finite values."
                )