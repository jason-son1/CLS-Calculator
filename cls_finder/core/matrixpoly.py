import numpy as np
from cls_finder.core.laurent import LaurentPoly

class MatrixPoly:
    """
    Represents a matrix of LaurentPoly elements of size R x C.
    """
    def __init__(self, data, d=None):
        """
        data: 2D list/array of LaurentPoly objects, or equivalent objects.
        """
        self.rows = len(data)
        self.cols = len(data[0]) if self.rows > 0 else 0
        
        # Determine dimension d
        if d is None:
            # Find first non-empty LaurentPoly
            found_d = 1
            for row in data:
                for val in row:
                    if isinstance(val, LaurentPoly):
                        found_d = val.d
                        break
            self.d = found_d
        else:
            self.d = d
            
        # Convert all elements to LaurentPoly
        self.data = []
        for r in range(self.rows):
            row_data = []
            for c in range(self.cols):
                val = data[r][c]
                if not isinstance(val, LaurentPoly):
                    val = LaurentPoly.constant(val, self.d)
                row_data.append(val)
            self.data.append(row_data)

    @classmethod
    def identity(cls, Q, d):
        data = [[LaurentPoly.zero(d) for _ in range(Q)] for _ in range(Q)]
        for i in range(Q):
            data[i][i] = LaurentPoly.constant(1.0, d)
        return cls(data, d)

    @classmethod
    def zero(cls, R, C, d):
        data = [[LaurentPoly.zero(d) for _ in range(C)] for _ in range(R)]
        return cls(data, d)

    def __add__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Dimension mismatch for addition: ({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})")
        if self.d != other.d:
            raise ValueError(f"Laurent dimension mismatch: {self.d} vs {other.d}")
        
        new_data = []
        for r in range(self.rows):
            new_row = []
            for c in range(self.cols):
                new_row.append(self.data[r][c] + other.data[r][c])
            new_data.append(new_row)
        return MatrixPoly(new_data, self.d)

    def __sub__(self, other):
        if self.rows != other.rows or self.cols != other.cols:
            raise ValueError(f"Dimension mismatch for subtraction: ({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})")
        if self.d != other.d:
            raise ValueError(f"Laurent dimension mismatch: {self.d} vs {other.d}")
            
        new_data = []
        for r in range(self.rows):
            new_row = []
            for c in range(self.cols):
                new_row.append(self.data[r][c] - other.data[r][c])
            new_data.append(new_row)
        return MatrixPoly(new_data, self.d)

    def __mul__(self, other):
        # Scalar multiplication
        if isinstance(other, (int, float, complex, LaurentPoly)):
            new_data = []
            for r in range(self.rows):
                new_row = []
                for c in range(self.cols):
                    new_row.append(self.data[r][c] * other)
                new_data.append(new_row)
            return MatrixPoly(new_data, self.d)
            
        # Matrix multiplication
        if not isinstance(other, MatrixPoly):
            raise TypeError("Multiplication only supported with scalar, LaurentPoly, or MatrixPoly")
            
        if self.cols != other.rows:
            raise ValueError(f"Dimension mismatch for multiplication: ({self.rows}x{self.cols}) vs ({other.rows}x{other.cols})")
        if self.d != other.d:
            raise ValueError(f"Laurent dimension mismatch: {self.d} vs {other.d}")
            
        new_data = [[LaurentPoly.zero(self.d) for _ in range(other.cols)] for _ in range(self.rows)]
        for r in range(self.rows):
            for c in range(other.cols):
                val = LaurentPoly.zero(self.d)
                for k in range(self.cols):
                    val = val + self.data[r][k] * other.data[k][c]
                new_data[r][c] = val
        return MatrixPoly(new_data, self.d)

    def __rmul__(self, other):
        if isinstance(other, (int, float, complex, LaurentPoly)):
            return self.__mul__(other)
        raise TypeError("Right multiplication only supported with scalars or LaurentPoly")

    def transpose(self):
        new_data = [[self.data[r][c] for r in range(self.rows)] for c in range(self.cols)]
        return MatrixPoly(new_data, self.d)

    def dagger(self):
        r"""Hermitian conjugate transpose: A^\dagger_{ij} = A_{ji}^\dagger"""
        new_data = [[self.data[r][c].conjugate() for r in range(self.rows)] for c in range(self.cols)]
        return MatrixPoly(new_data, self.d)

    def trace(self):
        if self.rows != self.cols:
            raise ValueError("Trace is only defined for square matrices")
        tr = LaurentPoly.zero(self.d)
        for i in range(self.rows):
            tr = tr + self.data[i][i]
        return tr

    def submatrix(self, row_to_remove, col_to_remove):
        """Returns the submatrix obtained by removing the specified row and column."""
        new_data = []
        for r in range(self.rows):
            if r == row_to_remove:
                continue
            new_row = []
            for c in range(self.cols):
                if c == col_to_remove:
                    continue
                new_row.append(self.data[r][c])
            new_data.append(new_row)
        return MatrixPoly(new_data, self.d)

    def slice_matrix(self, keep_axis, fixed_fractional_coords):
        """
        Slices the d-dimensional MatrixPoly to a 1D MatrixPoly along keep_axis,
        by fixing other axes at fractional coordinates.
        fixed_fractional_coords: dict mapping axis_index -> xi_val (float)
        """
        new_data = []
        for r in range(self.rows):
            new_row = []
            for c in range(self.cols):
                new_row.append(self.data[r][c].slice_poly(keep_axis, fixed_fractional_coords))
            new_data.append(new_row)
        return MatrixPoly(new_data, d=1)

    def evaluate(self, k_val, primitive_vectors):
        """Evaluate the matrix at momentum k_val, returning a complex numpy array."""
        out = np.zeros((self.rows, self.cols), dtype=complex)
        for r in range(self.rows):
            for c in range(self.cols):
                out[r, c] = self.data[r][c].evaluate(k_val, primitive_vectors)
        return out

    def evaluate_batch(self, k_vals, primitive_vectors):
        """Vectorized evaluation at N k-points. k_vals: (N, spatial_dim) → (N, rows, cols) complex."""
        k_vals = np.asarray(k_vals, dtype=float)
        N = len(k_vals)
        out = np.zeros((N, self.rows, self.cols), dtype=complex)
        for r in range(self.rows):
            for c in range(self.cols):
                out[:, r, c] = self.data[r][c].evaluate_batch(k_vals, primitive_vectors)
        return out

    def _total_terms(self):
        return sum(len(self.data[r][c].coefs)
                   for r in range(self.rows) for c in range(self.cols))

    def _mul_ops(self, other):
        """Projected term-operation count of self * other (sum of len*len over
        the contracted index), computed from term counts only — cheap O(rows*
        cols*k) integer work used to predict the cost of a multiply before
        paying for it."""
        ops = 0
        for i in range(self.rows):
            for j in range(other.cols):
                for k in range(self.cols):
                    ops += len(self.data[i][k].coefs) * len(other.data[k][j].coefs)
        return ops

    def det_and_adjugate(self, max_terms=None, max_seconds=None, max_ops=None):
        """
        Computes the determinant (LaurentPoly) and adjugate (MatrixPoly) of a square matrix
        using the Faddeev-LeVerrier algorithm (division-free Newton identities).

        max_terms / max_seconds / max_ops : optional cooperative budgets.
        Intermediate matrices' term count and runtime grow multiplicatively for
        dense, high-degree Laurent inputs. The budgets bound the cost so callers
        fall back gracefully instead of running (effectively) forever, and all
        work under Pyodide where CPU-bound Python cannot be interrupted:
          · max_ops     — projected term-operations of the *next* multiply; the
                          finest guard, since a single round can dominate runtime
                          and the per-round time check would never be reached.
          · max_terms   — term count of the running intermediate matrix.
          · max_seconds — wall-clock backstop, checked between rounds.
        Exceeding any of them raises MemoryError.
        """
        if self.rows != self.cols:
            raise ValueError("Determinant and adjugate are only defined for square matrices")

        Q = self.rows
        d = self.d

        import time as _time
        t0 = _time.perf_counter()

        # M_0 = I
        M = MatrixPoly.identity(Q, d)

        # List of M_r and coefficients c_r
        # c_r is a LaurentPoly
        c_list = [LaurentPoly.constant(1.0, d)]  # c_0 = 1 (placeholder)
        M_list = [M]

        for r in range(1, Q + 1):
            M_prev = M_list[r - 1]
            if max_terms is not None and M_prev._total_terms() > max_terms:
                raise MemoryError(
                    f"det_and_adjugate exceeded term budget ({max_terms}) at step {r}")
            if max_seconds is not None and (_time.perf_counter() - t0) > max_seconds:
                raise MemoryError(
                    f"det_and_adjugate exceeded time budget ({max_seconds}s) at step {r}")
            if max_ops is not None and self._mul_ops(M_prev) > max_ops:
                raise MemoryError(
                    f"det_and_adjugate projected step {r} exceeds op budget ({max_ops})")
            # A_M = A * M_{r-1}
            A_M = self * M_list[r - 1]
            # c_r = -Tr(A * M_{r-1}) / r
            c_r = A_M.trace() * (-1.0 / r)
            c_list.append(c_r)
            
            if r < Q:
                # M_r = A * M_{r-1} + c_r * I
                M_r = A_M + MatrixPoly.identity(Q, d) * c_r
                M_list.append(M_r)
        
        # det(A) = (-1)^Q * c_Q
        det_val = c_list[Q] * ((-1) ** Q)
        
        # adj(A) = (-1)^(Q-1) * M_{Q-1}
        adj_matrix = M_list[Q - 1] * ((-1) ** (Q - 1))
        
        return det_val, adj_matrix

    def is_zero(self, tol=1e-9):
        for r in range(self.rows):
            for c in range(self.cols):
                if not self.data[r][c].is_zero(tol):
                    return False
        return True

    def __repr__(self):
        lines = []
        for r in range(self.rows):
            row_str = " | ".join(str(self.data[r][c]) for c in range(self.cols))
            lines.append(f"[ {row_str} ]")
        return "\n".join(lines)
