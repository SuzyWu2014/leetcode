import math, sys

class Solution(object):

    def numSquares_four(self, n):
        """
        Lagrange's four_square theorem
        :param n:
        :return:
        """
        if n== 0: return 0
        while n % 4 == 0: n /= 4
        if n % 8 == 7: return 4
        m = int(n ** 0.5) + 1
        for x in xrange(m):
            y = int((n - x * x) ** 0.5)
            if x * x + y * y == n:
                if x > 0 and y > 0:
                    return 2
                else:
                    return 1
        return 3

    def numSquares_dp(self, n):  # TLE
        dp = [0 for _ in xrange(n + 1)]
        dp[1] = 1
        if n > 1:
            m = 2
            while m <= n:
                k = int(math.floor(math.sqrt(m)))
                minvalue = 5
                for i in xrange(1, k + 1):
                    minvalue = min(minvalue, dp[m - i * i])
                dp[m] = 1 + minvalue
                m += 1
        return dp[n]

if __name__ == "__main__":
    mysolution = Solution()
    res = mysolution.numSquares(5238)
    print res
