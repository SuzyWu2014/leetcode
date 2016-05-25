
class Solution(object):

    def lengthOfLIS_dp1(self, nums):  # O(n^2) time, O(n^2) space
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n== 0: return n

        # dp[i] denotes the length of LIS of nums[0..i]
        dp = [1 for _ in xrange(n)]

        for i in xrange(n):
            for j in xrange(i):
                if nums[j] < nums[i]:
                    dp[i] = max(dp[i], dp[j] + 1)
        return max(dp)

    def lengthOfLIS_dp2(self, nums):  # RT: O(nlogn)
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n == 0: return n

        # dp[i] denotes the index of the smallest one of the last elements of all i-length LISs
        dp = [0 for _ in xrange(n)]

        lengthOfLIS = 0
        for i in xrange(1, n):
            if nums[dp[0]] > nums[i]:
                # the index of the smallest one of the last elements of all LISs of length 1
                dp[0] = i
            elif nums[dp[lengthOfLIS]] < nums[i]:
                # increment current length of LIS by 1, if nums[i] is larger than the last element of current LIS
                lengthOfLIS += 1
                dp[lengthOfLIS] = i
            else:
                # if nums[i] is not larger than the last element of current LIS, then binary search for a shorter LIS,
                # of which nums[i] is larger than the last element and can increment the length by 1.
                index = self.getCeilIndex(nums, dp, lengthOfLIS, nums[i])
                dp[index] = i
        return lengthOfLIS + 1

    def getCeilIndex(self, nums, dp, length, val):  # binary search: O(logn)
        start = 0
        end = length
        while start <= end:
            mid = (start + end) / 2
            if mid < length and nums[dp[mid]] < val <= nums[dp[mid + 1]]:
                return mid + 1
            elif val < nums[dp[mid]]:
                end = mid - 1
            else:
                start = mid + 1
        return -1

if __name__ == "__main__":
    mysolution = Solution()
    res = mysolution.lengthOfLIS_dp2([10,9,2,5,3,7,101,18])
    print res

