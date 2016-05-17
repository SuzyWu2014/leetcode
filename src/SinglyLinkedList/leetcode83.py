

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        fast = head
        while fast != None and fast.next != None:
            if fast.val == fast.next.val:
                curr = fast
                while fast!=None and fast.val==curr.val:
                    fast = fast.next
                curr.next = fast
            else:
                fast = fast.next
        return head
