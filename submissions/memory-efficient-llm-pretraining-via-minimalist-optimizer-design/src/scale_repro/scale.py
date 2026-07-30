import math
from copy import deepcopy
from functools import reduce
from operator import mul


def column_normalize(matrix, eps=1e-12):
    """Normalize each matrix column by its L2 norm."""
    rows = _rectangular_matrix(matrix)
    column_count = len(rows[0])
    norms = [
        max(math.sqrt(sum(row[column] ** 2 for row in rows)), eps)
        for column in range(column_count)
    ]
    return [
        [value / norms[column] for column, value in enumerate(row)]
        for row in rows
    ]


def row_normalize(matrix, eps=1e-12):
    rows = _rectangular_matrix(matrix)
    normalized = []
    for row in rows:
        norm = max(math.sqrt(sum(value**2 for value in row)), eps)
        normalized.append([value / norm for value in row])
    return normalized


class ScaleToyOptimizer:
    """A deterministic CPU model of SCALE's state-allocation behavior."""

    def __init__(self, parameter_shapes, lm_output_parameter, beta=0.9):
        if lm_output_parameter not in parameter_shapes:
            raise ValueError("lm_output_parameter")
        if not 0.0 <= beta < 1.0:
            raise ValueError("beta")
        self.parameter_shapes = dict(parameter_shapes)
        self.lm_output_parameter = lm_output_parameter
        self.beta = beta
        self.state = {name: {} for name in self.parameter_shapes}
        self.last_updates = {}

    def step(self, gradients):
        missing = set(self.parameter_shapes) - set(gradients)
        if missing:
            raise ValueError(f"missing gradients: {sorted(missing)}")

        updates = {}
        for name, gradient in gradients.items():
            rows = _rectangular_matrix(gradient)
            if _shape(rows) != tuple(self.parameter_shapes[name]):
                raise ValueError(f"shape mismatch for {name}")

            if name == self.lm_output_parameter:
                previous = self.state[name].get("momentum", _zeros_like(rows))
                momentum = _matrix_add(
                    _matrix_scale(previous, self.beta),
                    _matrix_scale(rows, 1.0 - self.beta),
                )
                self.state[name]["momentum"] = momentum
                updates[name] = column_normalize(momentum)
            else:
                updates[name] = column_normalize(rows)

        self.last_updates = updates
        return deepcopy(updates)

    def momentum_parameter_names(self):
        return sorted(
            name for name, state in self.state.items() if "momentum" in state
        )


def memory_accounting(parameter_shapes, lm_output_parameter, bytes_per_value=4):
    if lm_output_parameter not in parameter_shapes:
        raise ValueError("lm_output_parameter")
    parameter_bytes = {
        name: _numel(shape) * bytes_per_value for name, shape in parameter_shapes.items()
    }
    full_momentum_bytes = sum(parameter_bytes.values())
    last_layer_momentum_bytes = parameter_bytes[lm_output_parameter]
    return {
        "bytes_per_value": bytes_per_value,
        "parameter_bytes": parameter_bytes,
        "full_momentum_bytes": full_momentum_bytes,
        "last_layer_momentum_bytes": last_layer_momentum_bytes,
        "last_layer_fraction_of_full_momentum": (
            last_layer_momentum_bytes / full_momentum_bytes
        ),
        "saved_momentum_bytes": full_momentum_bytes - last_layer_momentum_bytes,
    }


def _rectangular_matrix(matrix):
    rows = [[float(value) for value in row] for row in matrix]
    if not rows or not rows[0]:
        raise ValueError("matrix")
    width = len(rows[0])
    if any(len(row) != width for row in rows):
        raise ValueError("matrix")
    return rows


def _shape(matrix):
    return (len(matrix), len(matrix[0]))


def _zeros_like(matrix):
    return [[0.0 for _ in row] for row in matrix]


def _matrix_scale(matrix, scalar):
    return [[scalar * value for value in row] for row in matrix]


def _matrix_add(left, right):
    return [
        [left_value + right_value for left_value, right_value in zip(left_row, right_row)]
        for left_row, right_row in zip(left, right)
    ]


def _numel(shape):
    if not shape or any(int(dimension) <= 0 for dimension in shape):
        raise ValueError("shape")
    return int(reduce(mul, shape, 1))
