

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def deleteDuplicates(self, head):
        if head==None or head.next==None: return head
        dummy = ListNode(0)
        dummy.next = head
        pre, curr, next = dummy, head, head.next
        while next:
            if curr.val != next.val:
                next = next.next
                curr = curr.next
                pre = pre.next
            else:
                while next and next.val==curr.val:
                    next = next.next
                curr = next
                pre.next = curr
                if next: next = next.next
                else: break
        return dummy.next
