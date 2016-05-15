
class Solution(object):
    def findMin(self, nums): # RT: O(logn)
        n = len(nums)
        if nums[0]<=nums[-1]:
            return nums[0]
        start, end = 0, n-1
        while start <= end:
            mid = (start+end)/2
            if nums[mid] <= nums[end]:
                if mid > 0 and nums[mid-1]<nums[mid]:
                    end = mid-1
                else:
                    return nums[mid]
            else:
                start = mid+1
        return -1
