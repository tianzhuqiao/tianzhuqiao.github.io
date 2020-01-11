from enum import Enum
import numpy as np

class Munkres():
    """ https://pdfs.semanticscholar.org/848c/717ba51e48afef714dfef4bd6ab1cc050dab.pdf """
    class Steps(Enum):
        STEP1 = 1
        STEP2 = 2
        STEP3 = 3
        DONE = 4
    class Zeros(Enum):
        NORMAL = 0
        STAR = 1
        PRIME = 2

    def __init__(self, C, verbose=False):
        self.verbose = verbose
        # cost matrix
        self.C_orig = C
        self.C = C
        self.nrow, self.ncol = C.shape[0], C.shape[1]

        # mask matrix for zeros (normal, starred, primed)
        self.M = np.zeros_like(C).astype(int)
        self.M.fill(self.Zeros.NORMAL.value)
        # vector for row/column cover, 1 -> covered, 0 -> not covered
        self.row_cover = np.zeros(self.nrow).astype(int)
        self.col_cover = np.zeros(self.ncol).astype(int)
        self.uncovered_zero_prime = (0, 0)

        # result
        self.min_cost = None
        self.row_ind = None
        self.col_ind = None

    def run(self):
        ''' main loop '''
        step = self.step_zero()
        while True:
            if self.verbose:
                print("step:%d"%step.value)
                self.show_cost_matrix()
                self.show_mask_matrix()
            if step == self.Steps.STEP1:
                step = self.step_one()
            elif step == self.Steps.STEP2:
                step = self.step_two()
            elif step == self.Steps.STEP3:
                step = self.step_three()
            elif step == self.Steps.DONE:
                self.row_ind, self.col_ind = np.where(self.M == self.Zeros.STAR.value)
                self.min_cost = np.sum(self.C_orig[self.row_ind, self.col_ind])
                if self.verbose:
                    print("min cost: %f"%(self.min_cost))
                break

    def show_cost_matrix(self):
        ''' print current cost matrix '''
        for r in range(self.nrow):
            for c in range(self.ncol):
                print(self.C[r, c], end=" ")
            print("\n")

    def show_mask_matrix(self):
        ''' print mask (starred, primed zeros) matrix '''
        print(" ", end=" ")
        for c in range(self.ncol):
            print(int(self.col_cover[c]), end=" ")
        print('\n')
        for r in range(self.nrow):
            print(int(self.row_cover[r]), end=" ")
            for c in range(self.ncol):
                print(int(self.M[r, c]), end=" ")
            print('\n')

    def step_zero(self):
        """ preliminaries step """
        # subtract minimum from each row """
        self.C = self.C - self.C.min(axis=1, keepdims=True)

        # mark the starred zero in C; each row and col will have maximum 1 startted zero
        rows, cols = self.C.shape
        for r in range(rows):
            for c in range(cols):
                if self.C[r, c] == 0 and self.row_cover[r] == 0 and self.col_cover[c] == 0:
                    self.M[r, c] = self.Zeros.STAR.value
                    self.row_cover[r] = 1
                    self.col_cover[c] = 1

        return self.cover_star()

    def cover_star(self):
        """ cover each cololumn containing a starred zero """
        # clear covers
        self.row_cover.fill(0)
        self.col_cover.fill(0)
        # clear primed zeros
        self.M[self.M == self.Zeros.PRIME.value] = 0

        # find the starred zeros
        star = self.M == self.Zeros.STAR.value
        # cover each column that containing a starred zero
        self.col_cover = (star.sum(axis=0) > 0).astype(int)

        # calculated the number of covered cols
        colcount = self.col_cover.sum()
        if(colcount >= self.ncol or colcount >= self.nrow):
            # done
            return self.Steps.DONE

        return self.Steps.STEP1

    def step_one(self):
        """ find uncovered zeros and prime it """
        C_zeros = (self.C == 0).astype(int)
        C_zeros_uncovered = C_zeros * (1-self.row_cover[:, np.newaxis])
        C_zeros_uncovered *= (1-self.col_cover)

        while True:
            # find a uncovered zero
            # looks like np.argmax is fast than np.nozero, np.where
            row, col = np.unravel_index(np.argmax(C_zeros_uncovered), C_zeros_uncovered.shape)
            if C_zeros_uncovered[row, col] == 0:
                # no uncovered zeros
                return self.Steps.STEP3

            # prime it
            self.M[row, col] = self.Zeros.PRIME.value
            if self.star_in_row(row):
                # star in this row,
                col = self.find_star_in_row(row)
                # cover row
                self.row_cover[row] = 1
                # uncover the column
                self.col_cover[col] = 0
                C_zeros_uncovered[:, col] = C_zeros[:, col]*(1-self.row_cover)
                C_zeros_uncovered[row] = 0
            else:
                self.uncovered_zero_prime = (row, col)
                return self.Steps.STEP2

    def star_in_row(self, row):
        ''' check if there is a starred zero in row '''
        return np.any(self.M[row, :] == self.Zeros.STAR.value)

    def find_star_in_row(self, row):
        ''' find the col index of starred zero in row '''
        rst = np.where(self.M[row, :] == self.Zeros.STAR.value)[0]
        if rst.size:
            return rst[0]
        return None

    def step_two(self):
        ''' Find a better cover from current zeros '''
        # construct a sequence from the uncovered primed zero from step 1
        path_count = 1
        path = np.zeros((self.nrow+self.ncol, 2)).astype(int)
        path[path_count - 1, :] = self.uncovered_zero_prime

        while True:
            r = self.find_star_in_col(path[path_count-1, 1])
            if r is None:
                break

            path_count += 1
            path[path_count-1, :] = [r, path[path_count - 2, 1]]

            c = self.find_prime_in_row(path[path_count - 1, 0])
            path_count += 1
            path[path_count-1, :] = [path[path_count - 2, 0], c]

        # unstar the starred zeros, and star the primed zeros in the sequence
        for p in range(path_count):
            if self.M[path[p, 0], path[p, 1]] == self.Zeros.STAR.value:
                self.M[path[p, 0], path[p, 1]] = self.Zeros.NORMAL.value
            else:
                self.M[path[p, 0], path[p, 1]] = self.Zeros.STAR.value
        # cover the starred columns
        return self.cover_star()

    def find_star_in_col(self, col):
        ''' find the row index of starred zero in col '''
        rst = np.where(self.M[:, col] == self.Zeros.STAR.value)[0]
        if rst.size:
            return rst[0]
        return None

    def find_prime_in_row(self, row):
        ''' find the col index of primed zero in row '''
        rst = np.where(self.M[row, :] == self.Zeros.PRIME.value)[0]
        if rst.size:
            return rst[0]
        return None

    def step_three(self):
        """
        subtract smallest value from uncovered cells to each uncovered cols, and
        add it to each covered rows. so the final result is always positive
        """
        # find the smallest value from uncovered cells
        minval = self.find_smallest()
        # add it to covered rows
        self.C += self.row_cover[:, np.newaxis]*minval
        # subtract it from uncovered columns
        self.C -= (1-self.col_cover)*minval
        return self.Steps.STEP1

    def find_smallest(self):
        """ find the smallest value from uncovered cells """
        # add max value to covered rows and columns to ignore the covered cells
        maxval = self.C.max()
        C = self.C + self.row_cover[:, np.newaxis]*maxval
        C += self.col_cover*maxval
        # return the smallest value
        return C.min()

if __name__ == '__main__':
    import time
    C = np.array([1, 2, 3, 2, 4, 6, 3, 6, 9]).reshape(3, 3)
    C = np.array([1, 2, 3, 4, 2, 4, 6, 8, 3, 6, 9, 12, 4, 8, 12, 16]).reshape(4, 4).astype(float)
    C = np.array([1, 1, 3, 2, 4, 6, 3, 6, 9]).reshape(3,3).astype(float)
    #C = np.abs(np.random.rand(500, 500))
    start = time.time()
    m = Munkres(C=C, verbose=False)
    m.run()
    end = time.time()
    print("time: %.4f s"%(end-start))

    from scipy.optimize import linear_sum_assignment
    start = time.time()
    t = linear_sum_assignment(C)
    end = time.time()
    print("time: %.4f s"%(end-start))
    print(np.sum(C[t[0], t[1]]), m.min_cost)
    assert sum(np.abs(m.col_ind - t[1] == 0))
