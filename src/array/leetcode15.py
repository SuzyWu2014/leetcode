
class Solution(object):
    def threeSum_dfs(self, nums): # DFS, but TLE
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def dfs(depth, target, nums, valuelist):
            if depth == 0:
                if target == 0: res.add(tuple(valuelist))
                return
            if target<nums[0] or target>nums[len(nums)-1]: return
            for i in xrange(len(nums)-depth+1):
                dfs(depth - 1, target - nums[i], nums[i + 1:], valuelist + [nums[i]])
        if nums == []: return []
        nums.sort()
        res = set()
        dfs(3, 0, nums, [])
        return [list(t) for t in res]

    def threeSum_Generic(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        def ksum(nums,k,target):
            # This is a generic k-sum algorithm
            res = set() # avoid duplicates
            i=0
            if k==2:
                j = len(nums)-1
                while i<j:
                    if nums[i]+nums[j]==target:
                        res.add((nums[i],nums[j]))
                        i+=1
                    elif nums[i]+nums[j]>target:
                        j-=1
                    else:
                        i+=1
            else:   # case: k>2
                for i in xrange(len(nums)-k+1):
                    newtarget = target - nums[i]
                    subresult = ksum(nums[i+1:], k-1, newtarget)
                    if subresult:
                        res = res | set((nums[i],)+nr for nr in subresult)
            return res

        nums.sort() # O(nlog(n))
        return [list(t) for t in ksum(nums, 3, 0)]
