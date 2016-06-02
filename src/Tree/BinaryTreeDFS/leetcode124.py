
# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxPathSum(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        def dfs(root):
            if root is None: return 0
            sum = root.val
            lmax = rmax = 0
            if root.left:
                lmax = dfs(root.left)
                if lmax > 0: sum += lmax
            if root.right:
                rmax = dfs(root.right)
                if rmax > 0: sum += rmax
            self.max = max(self.max, sum)
            return max(root.val, root.val + lmax, root.val + rmax)

        if root is None: return 0
        self.max = -10000000
        dfs(root)
        return self.max
