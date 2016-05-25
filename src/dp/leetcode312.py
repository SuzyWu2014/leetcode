

class Solution(object):
    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        newnums = [1] + nums + [1]
        n = len(newnums)
        dp = [[0]*n for _ in xrange(n)]
        # the length of each span
        for length in xrange(2, n):
            # the left bound of the range to check
            for start in range(n-length):
                # the right bound of the range to check
                end = start+length
                for lastBalloon in xrange(start+1, end):
                    dp[start][end] = max(dp[start][end], dp[start][lastBalloon]+newnums[start]*newnums[lastBalloon]*newnums[end]+dp[lastBalloon][end])
        return dp[0][n-1]
