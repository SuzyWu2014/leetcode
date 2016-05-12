
class Solution(object):
    def setZeroes(self, matrix):    # O(mn) time, O(m+n) space
        """
        :type matrix: List[List[int]]
        :rtype: void Do not return anything, modify matrix in-place instead.
        """
        m,n = len(matrix), len(matrix[0])
        zerorows = [False for _ in xrange(m)]
        zerocols = [False for _ in xrange(n)]

        for x in xrange(m):
            for y in xrange(n):
                if matrix[x][y] == 0:
                    zerorows[x], zerocols[y] = True, True

        for x in xrange(m):
            for y in xrange(n):
                if zerorows[x] or zerocols[y]:
                    matrix[x][y] = 0