
class Solution(object):
    def trap(self, height):  # O(n) time, O(n) space
        """
        :type height: List[int]
        :rtype: int
        """
        n = len(height)
        if n==0: return 0
        # from left to right, compute leftmost[i],
        # which means the most height on the left
        # side of the i-th position
        leftmost = [height[0]]
        for x in xrange(1, n):
            leftmost.append(max(leftmost[-1], height[x]))
        water = 0
        # rightmost indicates the most height on the
        # right side of the i-th position from right
        # to left
        rightmost = height[n-1]
        for x in xrange(n-1, -1, -1):
            if height[x]<rightmost and height[x]<leftmost[x]:
                water += min(rightmost, leftmost[x]) - height[x]
            elif height[x] > rightmost:
                rightmost = height[x]
        return water
