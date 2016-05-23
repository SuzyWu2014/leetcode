

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None


class Solution(object):

    def inorderTraversal_iterative(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if root is None:
            return res

        stack = []
        while len(stack) > 0 or root:
            if root:
                stack.append(root)
                root = root.left
            else:
                root = stack.pop()
                res.append(root.val)
                root = root.right
        return res

    def inorderTraversal_recursive(self, root):
        """
        :type root: TreeNode
        :rtype: List[int]
        """
        res = []
        if root:
            return res

        # traverse left subtree
        if root.left:
            res += self.inorderTraversal_recursive(root.left)

        # visit root
        res.append(root.val)

        # traverse right subtree
        if root.right:
            res += self.inorderTraversal_recursive(root.right)

        return res