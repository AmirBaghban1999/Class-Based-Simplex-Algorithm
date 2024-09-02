import numpy as np

class MatrixOperations:
    def __init__(self, matrix):
        self.matrix = np.array(matrix, dtype=float)

    def inversematrix(self):
        """
        Computes the inverse of a matrix using a manual row operation method.
        """
        a = self.matrix.copy()
        counter = []
        n = len(a[0])
        inverse = np.empty((n, n), dtype=float)
        c = np.empty((n, 2 * n), dtype=float)
        b = np.empty((n, n + n), dtype=float)
        identity = np.eye(n, dtype=float)

        for k in range(n):
            pp = list(a[k])
            for t in range(n):
                pp.append(identity[k, t])
            b[k] = np.array(pp)

        for j in range(n):
            for i in range(n):
                if b[j, i] != 0:
                    b[j] = (1 / b[j][i]) * b[j]
                    counter.append((j, i))
                    break
            for t in range(n):
                if t != j and b[t, i] != 0:
                    b[t] = b[t] - b[t][i] * b[j]

        for k in counter:
            c[k[0]] = b[k[1]]
        for i in range(n):
            inverse[i] = c[i, n:]

        return inverse

# Example Usage:
a = np.array([
    [0, 3, 1, 0],
    [8, 9, 1, 1],
    [3, 2, 10, 0],
    [0, 0, 0, 1]
], dtype=float)

matrix_ops = MatrixOperations(a)
inverse = matrix_ops.inversematrix()

print(inverse)