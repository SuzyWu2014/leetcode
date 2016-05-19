

# Definition for singly-linked list with a random pointer.
class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None


class Solution(object):
    def copyRandomList(self, head): # RT: O(n)
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        if head==None: return head

        # copy every existed node in the original
        # list and insert the copy node into the list
        # just next to the original node.
        # before: 1->2->3->4
        # after:  1->1->2->2->3->4->4
        p = head
        while p!=None:
            node = RandomListNode(p.label)
            tmp = p.next
            node.next = tmp
            p.next = node
            p = node.next

        # deal with the random pointers
        p = head
        while p!=None:
            if p.random!=None:
                p.next.random = p.random.next
            p = p.next.next

        # separate the copy list from the
        # the hybrid list, and recover the
        # original list
        dummy = RandomListNode(0)
        dummy.next = head.next
        q = dummy
        p = head
        while p!=None:
            q.next = p.next
            p.next = p.next.next
            q = q.next
            p = p.next

        return dummy.next
