
import heapq

# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class Solution(object):
    def mergeKLists(self, lists): # use min heap
        """
        :type lists: List[ListNode]
        :rtype: ListNode
        """
        heap = []
        for alist in lists:
            if alist:
                # push the value of first node of each list and
                # the list itself as a tuple into heap
                heap.append((alist.val, alist))
        # transform 'heap' of list type into a heap, in-place, in linear time.
        heapq.heapify(heap)
        dummy = ListNode(0)
        curr = dummy
        while heap:
            val, alist = heapq.heappop(heap)
            curr.next = ListNode(val)
            curr = curr.next
            if alist.next:
                heapq.heappush(heap, (alist.next.val, alist.next))
        return dummy.next
