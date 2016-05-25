
class Solution(object):
    def rob(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums is None or len(nums) == 0: return 0
        elif len(nums) == 1: return nums[0]
        # As the houses form a circle, if rob the 1st house,
        # you cannot rob the last one, so need 2 dp scan.
        return max(self.roblinear(nums[:len(nums ) -1]), self.roblinear(nums[1:]))

    def roblinear(self, nums):
        n = len(nums)
        if n == 1: return nums[0]
        dp = [nums[0]]
        dp.append(max(nums[1], dp[0]))
        for i in range(2, n):
            dp.append(max(dp[i - 1], dp[i - 2] + nums[i]))
        return dp[n - 1]
