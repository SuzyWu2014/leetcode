

class Solution(object):
    def generateParenthesis(self, n):
        """
        :type n: int
        :rtype: List[str]
        """
        def dfs(left, right, valuelist):
            if left<0 or right<0:
                return
            if left==right==0:
                res.append(valuelist)
            elif left==right:
                dfs(left-1, right, valuelist+'(')
            elif left<right:
                dfs(left-1, right, valuelist+'(')
                dfs(left, right-1, valuelist+')')

        res = []
        dfs(n,n,'')
        return res
