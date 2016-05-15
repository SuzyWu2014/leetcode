
class Solution(object):
    def jump(self, nums):
        n = len(nums)
        if n == 0 or n == 1:
            return 0

        jumps = lastCanReach = currCanReach = 0
        for i in xrange(n):
            if i == 0:
                lastCanReach = nums[0]
                jumps = 1
            else:
                if i <= lastCanReach:
                    currCanReach = max(currCanReach, nums[i] + i)
                    if i == lastCanReach and currCanReach > lastCanReach:
                        lastCanReach = currCanReach
                        jumps += 1
            if lastCanReach >= n - 1: break
        return jumps

