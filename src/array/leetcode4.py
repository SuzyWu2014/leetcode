
class Solution(object):
    # use binary search idea: each time removes k/2 elements at most
    # RT: O(log(m+n)), m is the length of nums, n is the
    # length of nums2
    def findMedianSortedArrays(self, nums1, nums2):
        """
        :type nums1: List[int]
        :type nums2: List[int]
        :rtype: float
        """
        total = len(nums1) + len(nums2)
        if total % 2 != 0:  # total is odd
            return self.find_kth(nums1, nums2, total/2+1)
        else:
            return (self.find_kth(nums1, nums2, total/2) + self.find_kth(nums1, nums2, total/2+1)) / 2.0

    def find_kth(self, nums1, nums2, k):
        # always assume the length of nums1 is less than that of nums2
        if len(nums1) > len(nums2):
            return self.find_kth(nums2, nums1, k)

        if len(nums1) == 0:
            return nums2[k-1]
        if k == 1:
            return min(nums1[0], nums2[0])

        n1 = min(k/2, len(nums1))
        n2 = k - n1
        if nums1[n1-1] < nums2[n2-1]:
            return self.find_kth(nums1[n1:], nums2, k-n1)
        elif nums1[n1-1] > nums2[n2-1]:
            return self.find_kth(nums1, nums2[n2:], k-n2)
        else:
            return nums1[n1-1]