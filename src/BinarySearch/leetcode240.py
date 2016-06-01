
class Solution(object):
    def searchMatrix_iterative(self, matrix, target): # RT: O(m+n)
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        m ,n = len(matrix), len(matrix[0])
        y = n-1
        for x in xrange(m):
            while y >= 0 and matrix[x][y] > target:
                y -= 1
            if matrix[x][y] == target:
                return True
        return False

    def searchMatrix_DivideConquer(self, matrix, target):  # RT: O(n^1.58)
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        def helper(matrix, rowStart, rowEnd, colStart, colEnd, target):
            if rowStart > rowEnd or colStart > colEnd:
                return False

            rowMid = (rowStart+rowEnd)/2
            colMid = (colStart+colEnd)/2

            if matrix[rowMid][colMid] > target:
                return helper(matrix, rowStart, rowMid-1, colStart, colMid-1, target) or \
                helper(matrix, rowMid, rowEnd, colStart, colMid-1, target) or \
                helper(matrix, rowStart, rowMid-1, colMid, colEnd, target)

            elif matrix[rowMid][colMid] < target:
                return helper(matrix, rowMid+1, rowEnd, colMid+1, colEnd, target) or \
                helper(matrix, rowMid+1, rowEnd, colStart, colMid, target) or \
                helper(matrix, rowStart, rowMid, colMid+1, colEnd, target)

            else:
                return True

        m, n = len(matrix), len(matrix[0])
        return helper(matrix, 0, m-1, 0, n-1, target)
