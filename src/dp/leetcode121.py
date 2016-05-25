
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n== 0: return 0
        dp = [0 for _ in xrange(n)]
        minprice = prices[0]
        for x in xrange(1, n):
            if prices[x] > minprice:
                dp[x] = max(prices[x] - minprice, dp[x - 1])
            else:
                minprice = prices[x]
                dp[x] = dp[x - 1]
        return dp[n-1]
