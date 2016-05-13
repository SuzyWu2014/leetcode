
class Solution(object):
    def productExceptSelf(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        n = len(nums)
        res = [1]*n
        product = 1
        for x in xrange(n-1):
            product *= nums[x]
            res[x+1] *= product
        product = 1
        for x in xrange(n-1, 0, -1):
            product *= nums[x]
            res[x-1] *= product
        return res
