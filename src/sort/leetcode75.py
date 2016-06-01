

class Solution(object):
    def sortColors(self, nums):
        """
        :type nums: List[int]
        :rtype: void Do not return anything, modify nums in-place instead.
        """
        if len(nums) > 0:
            red = white = blue = 0
            for num in nums:
                if num == 0:
                    red += 1
                elif num == 1:
                    white += 1
                elif num == 2:
                    blue += 1
            for i in xrange(len(nums)):
                if i < red:
                    nums[i] = 0
                elif i < red+white:
                    nums[i] = 1
                elif i < red+white+blue:
                    nums[i] = 2
