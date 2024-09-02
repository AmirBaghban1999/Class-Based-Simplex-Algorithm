import numpy as np
from matrix_operations import MatrixOperations

class SimplexAlgorithm:
    '''
A: This is the matrix of coefficients for the constraints in the linear programming problem. Each row in A corresponds to a constraint, and each column corresponds to a variable in the linear program.

C: This is the cost function's coefficients vector. It represents the coefficients of the objective function, which you want to maximize or minimize. If the objective function is of the form c1*x1 + c2*x2 + ..., then C = [c1, c2, ...].

b: This is the right-hand side vector for the constraints. Each element in b corresponds to the right-hand side of a constraint in the problem, i.e., the constants on the right side of each equation in the system AX = b.

IB: This is the list of indices that correspond to the basis columns in matrix A. These indices indicate which variables are currently in the basis.

IN: This is the list of indices that correspond to the non-basis columns in matrix A. These indices indicate which variables are currently non-basic.    
    '''
    def __init__(self, A, C, b, IB, IN):
        self.A = np.array(A)
        self.C = np.array(C)
        self.b = np.array(b)
        self.IB = list(IB)
        self.IN = list(IN)
        self.n = len(self.IB)
        self.m = len(self.A[0])
        self.B = np.zeros((self.n, self.n))
        self.N = np.zeros((self.n, self.m - self.n))
        self.CN = []
        self.CB = []
        self.matrix_ops = MatrixOperations(self.B)

    def update_matrices(self):
        for i in range(self.n):
            self.B[:, i] = self.A[:, self.IB[i]]
        Binverse = self.matrix_ops.inversematrix()

        self.CN = [self.C[col] for col in self.IN]
        self.CB = [self.C[col] for col in self.IB]

        for i in range(self.m - self.n):
            self.N[:, i] = self.A[:, self.IN[i]]

        CN = np.transpose(self.CN) - np.transpose(self.CB).dot(Binverse.dot(self.N))
        self.C = np.array([0 if i in self.IB else CN[self.IN.index(i)] for i in range(self.m)])

    def simplex(self):
        self.update_matrices()

        negative = [c for c in self.C if c < 0]
        self.A = self.matrix_ops.inversematrix().dot(self.A)
        self.b = self.matrix_ops.inversematrix().dot(self.b)

        if not negative:
            return self.IB, self.b

        while negative:
            enter = self.C.tolist().index(negative[0])
            enter_column = self.A[:, enter]

            if all(val <= 0 for val in enter_column):
                return "The problem is unbounded"

            min_selecting_vector = [i / j for i, j in zip(self.b, enter_column) if j > 0]
            min_value = min(min_selecting_vector)
            exit_index = list(self.b).index(next(i for i, j in zip(self.b, enter_column) if i == j * min_value))

            s = self.IB[exit_index]
            self.IB[exit_index] = enter
            r = self.IN.index(enter)
            self.IN[r] = s

            self.update_matrices()
            negative = [c for c in self.C if c < 0]

        return self.IB, self.b

# Example usage:
if __name__ == "__main__":
    A = np.array([[2, 3, 1, 0], [2, 1, 0, 1]])
    C = [-3, -2, 0, 0]
    b = [12, 8]
    IB = [2, 3]
    IN = [0, 1]

    simplex_solver = SimplexAlgorithm(A, C, b, IB, IN)
    result = simplex_solver.simplex()

    print(result)
