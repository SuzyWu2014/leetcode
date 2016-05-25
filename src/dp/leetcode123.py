
class Solution(object):
    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n== 0: return 0
        # the profits gained in the first i days
        profits1 = [0 for _ in range(n)]
        # the profits gained in the last i days
        profits2 = [0 for _ in range(n)]

        # compute the profits gained in the first transaction
        minprice = prices[0]
        profits1[0] = 0
        for i in range(1, n):
            profits1[i] = max(profits1[i - 1], prices[i] - minprice)
            minprice = min(minprice, prices[i])

        # compute the profits gained in the second transaction
        maxprice = prices[n - 1]
        profits2[n - 1] = 0
        for i in range(n - 2, -1, -1):
            profits2[i] = max(profits2[i + 1], maxprice - prices[i])
            maxprice = max(maxprice, prices[i])

        # merge the profits gained in these two transactions
        maxprofit = 0
        for i in range(n):
            maxprofit = max(maxprofit, profits1[i] + profits2[i])
        return maxprofit
