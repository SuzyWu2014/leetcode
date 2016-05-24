
class Solution(object):
    def nthUglyNumber_dp(self, n):  # TLE on OJ, O(n) time, O(n) space
        """
        :type n: int
        :rtype: int
        """
        dp = [False]
        count = 0;
        i = 0
        while count < n:
            i += 1
            if i in [1, 2, 3, 5]:
                dp.append(True)
            elif i % 2 == 0:
                dp.append(dp[i / 2])
            elif i % 3 == 0:
                dp.append(dp[i / 3])
            elif i % 5 == 0:
                dp.append(dp[i / 5])
            else:
                dp.append(False)

            if dp[i] == True:
                count += 1

        return i

    def nthUglyNumber(self, n):  # O(n) time, O(n) space
        """
        :type n: int
        :rtype: int
        """
        if n== 0: return 0
        if n == 1: return 1
        target = 1
        x2, x3, x5 = [], [], []
        for i in range(n - 1):
            x2.append(target * 2)
            x3.append(target * 3)
            x5.append(target * 5)
            target = min(x2[0], x3[0], x5[0])

            # remove the duplicates
            if target == x2[0]: x2.pop(0)
            if target == x3[0]: x3.pop(0)
            if target == x5[0]: x5.pop(0)
        return target

