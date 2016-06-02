

# Definition for a binary tree node.
# class TreeNode(object):
#     def __init__(self, x):
#         self.val = x
#         self.left = None
#         self.right = None

class Solution(object):
    def maxDepth(self, root):
        """
        :type root: TreeNode
        :rtype: int
        """
        if root is None:
            return 0
        # base case
        if root.left is None and root.right is None:
            return 1
        # recursive step
        elif root.left is None:
            return self.maxDepth(root.right)+1
        elif root.right is None:
            return self.maxDepth(root.left)+1
        else:
            return max(self.maxDepth(root.left), self.maxDepth(root.right))+1
