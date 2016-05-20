

class Solution(object):

    def strStr_Z(self, haystack, needle):  # O(m+n) time, O(m+n) space
        """
        Z-algorithm: text = needle + '$' + haystack
        """
        m, n = len(haystack), len(needle)
        if n == 0: return 0
        if m < n: return -1

        text = needle + '$' + haystack
        size = m + n + 1

        Z = self.calculate_Z(text)

        res = [i-n-1 for i in xrange(size) if Z[i] == n]

        if res == []:
            return -1
        else:
            return res[0]

    def calculate_Z(self, text):
        size = len(text)
        Z = [0 for _ in xrange(size)]
        # left bound and right bound
        left, right = 0, 0

        for k in xrange(1, size):
            if k == 13:
                pass
            if k > right:
                left = right = k
                while right < len(text) and text[right] == text[right-left]:
                    right += 1
                Z[k] = right-left
                right -= 1
            else:
                # operate inside box
                k1 = k - left
                # if value does not stretches till right bound, then just copy it
                if k+Z[k1] <= right:
                    Z[k] = Z[k1]
                else:
                    # try to see if there are more matches
                    left = k
                    while right < size and text[right] == text[right-left]:
                        right += 1
                    Z[k] = right-left
                    right -= 1
        return Z


if __name__ == "__main__":
    mysolution = Solution()
    result = mysolution.calculate_Z("aabxaabxcaabxaabxay")

