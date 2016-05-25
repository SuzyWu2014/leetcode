
class Solution(object):
    def coinChange_dp1(self, coins, amount):  # O(n^2) time, O(n) space
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        coins.sort()
        if amount== 0: return 0
        if amount < coins[0]: return -1
        # dp[i] denotes the minimum number of coins consisting of amount of i
        dp = [0] + [-1 for _ in xrange(amount)]
        candidates = []
        for c in coins:
            if c <= amount:
                dp[c] = 1
                candidates.append(c)
        for i in xrange(candidates[0] + 1, amount + 1):
            if i not in coins:
                minvalue = -1
                for coin in candidates:
                    if i >= coin and dp[i - coin] != -1:
                        if minvalue == -1:
                            minvalue = dp[i - coin]
                        else:
                            minvalue = min(minvalue, dp[i - coin])
                dp[i] = 1 + minvalue if minvalue > 0 else minvalue
        return dp[-1]

    def coinChange_dp2(self, coins, amount):  # O(n^2) time, O(n) space
        """
        :type coins: List[int]
        :type amount: int
        :rtype: int
        """
        # dp[i] denotes the minimum number of coins consisting of amount of i
        dp = [0] + [-1] * amount
        for x in range(amount):
            if dp[x] < 0: continue
            for c in coins:
                if x + c > amount:
                    continue
                if dp[x + c] < 0 or dp[x + c] > dp[x] + 1:
                    dp[x + c] = dp[x] + 1
        return dp[amount]
