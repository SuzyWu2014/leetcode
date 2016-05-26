

class Solution(object):
    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(t, start, valuelist):
            # base case
            if t==0:
                return res.append(valuelist)

            # recursive step
            for i in xrange(start, len(candidates)):
                if candidates[i]<=t:
                    dfs(t-candidates[i], i, valuelist+[candidates[i]])
        # sorting for monotony of the resulted sequence and optimizing the candidate pool
        candidates.sort()
        res = []
        dfs(target, 0, [])
        return res
