"""Formulation selection, independent of how its material tangent is evaluated.

The original residual/Jacobian pair lives in alternatives/original/.
The coupled formulation uses alternatives/coupled/;
its strain unknowns and complementary face feedback are solved monolithically.
The retained formulation uses core/ and solves increments in
cell fields and free-face displacement, with nested constitutive updates.
"""

from typing import Literal, TypeAlias

FormulationName: TypeAlias = Literal["original", "coupled", "retained"]
DEFAULT_FORMULATION: FormulationName = "retained"
FORMULATION_CHOICES: tuple[FormulationName, ...] = ("retained", "original", "coupled")


def select_formulation(name: str) -> FormulationName:
    """Validate a selection before preparing a case, solving, or writing output.

    Each residual uses its matching analytical or finite-difference tangent.
    Unknown names never fall back to a different set of equations.
    """
    for formulation in FORMULATION_CHOICES:
        if name == formulation:
            return formulation
    raise ValueError("formulation must be 'original', 'coupled', or 'retained'.")
