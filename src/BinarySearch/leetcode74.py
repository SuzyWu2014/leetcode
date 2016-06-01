

class Solution(object):
    def searchMatrix(self, matrix, target): # RT: O(log(m+n))
        """
        :type matrix: List[List[int]]
        :type target: int
        :rtype: bool
        """
        m, n = len(matrix), len(matrix[0])
        start, end = 0, m*n-1
        while start <= end:
            mid = (start+end)/2
            row, col = mid/n, mid%n
            if matrix[row][col] < target:
                start = mid+1
            elif matrix[row][col] > target:
                end = mid-1
            else:
                return True
        return False
