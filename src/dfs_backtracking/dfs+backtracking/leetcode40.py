
class Solution(object):
    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(nums, t, start, valuelist):
            if t== 0:
                if valuelist not in res:
                    return res.append(valuelist)
            else:
                for i in xrange(start, len(nums)):
                    if t < nums[i]: return
                    dfs(nums, t-nums[i], i+1, valuelist + [nums[i]])

        candidates.sort()
        res = []
        dfs(candidates, target, 0, [])
        return res
