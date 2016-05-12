
class Solution(object):
    def rotate(self, matrix):   # O(n^2) time
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        rows = len(matrix)
        cols = len(matrix[0])

        if rows<2 or cols<2:
            return

        # invert the matrix
        for j in range(cols):
            for i in range(rows/2):
                matrix[i][j], matrix[rows-1-i][j] = matrix[rows-1-i][j], matrix[i][j]

        # flap the matrix along the main diagonal
        k = 0
        for i in range(rows):
            for j in range(k,cols):
                if i!=j:
                    matrix[i][j], matrix[j][i] = matrix[j][i], matrix[i][j]
            k += 1
