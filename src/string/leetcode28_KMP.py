

class Solution(object):

    def strStr_KMP(self, haystack, needle):  # O(m+n) time, O(n) space
        m, n = len(haystack), len(needle)
        if n == 0: return 0
        if m < n: return -1

        # traverse needle to fill out array T in O(n) time
        T = [0 for _ in xrange(n)]
        i, j = 1, 0
        while i < n:
            if needle[j] == needle[i]:
                T[i] = j + 1
                i += 1
                j += 1
            else:
                if j == 0:
                    T[i] = 0
                    i += 1
                else:
                    j = T[j-1]

        # traverse haystack in O(m) time
        i = j = 0
        while i < m and j < n:
            if haystack[i] == needle[j]:
                i += 1
                j += 1
            else:
                if j > 0:
                    j = T[j-1]
                else:
                    i += 1
        if j == n:
            return i - n
        else:
            return -1


if __name__ == "__main__":
    mysolution = Solution()
    result = mysolution.strStr_KMP("babba", "bbb")
