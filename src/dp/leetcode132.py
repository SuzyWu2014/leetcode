

class Solution(object):
    def minCut_dp1(self, s):
        """
        :type s: str
        :rtype: int
        """
        # the min cuts of the string s[i:]
        dp = [0 for i in range(len(s) + 1)]
        # p[i][j] is True, if s[i,j] is palindromic; False, otherwise.
        p = [[False for i in range(len(s))] for j in range(len(s))]
        # set dp[i] the upper bound
        for i in range(len(s) + 1):
            dp[i] = len(s) - i
        for i in range(len(s) - 1, -1, -1):
            for j in range(i, len(s)):
                # case1: i==j
                # case2: i==j+1, and s[i]==s[j]
                # case3: s[i]==s[j], and s[i+1...j-1] is palindromic
                if i == j or (s[i] == s[j] and i + 1 == j) or (s[i] == s[j] and p[i + 1][j - 1]):
                    p[i][j] = True
                    dp[i] = min(1 + dp[j + 1], dp[i])
        #return dp[0] - 1, p
        return dp

    def minCut_dp2(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)

        # T[i][j] is True, if s[i,j] is palindromic; False, otherwise.
        T = [[False] * n for _ in xrange(n)]
        for l in xrange(1, n + 1):
            for i in xrange(n - l + 1):
                j = i + l - 1
                if i == j:  # length of s[i,j] is 1
                    T[i][j] = True
                elif i == j - 1:  # length of s[i,j] is 2
                    T[i][j] = s[i] == s[j]
                else:
                    T[i][j] = (s[i] == s[j]) and T[i + 1][j - 1]

        cuts = [x for x in xrange(n)]
        for i in xrange(1, n):
            if s[0] == s[i] and (i == 1 or T[1][i - 1] == True):
                cuts[i] = 0
            else:
                j = 1
                while j < i:
                    if s[j] == s[i] and (j + 1 == i or T[j + 1][i - 1] == True):
                        cuts[i] = 1 + cuts[j - 1]
                        break
                    j += 1
                if j == i: cuts[i] = 1 + cuts[j - 1]
                cuts[i] = min(cuts[i], cuts[i - 1] + 1)
        # return cuts[-1]
        return cuts


if __name__ == "__main__":
    mysolution = Solution()
    s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaabbaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
    dp = mysolution.minCut_dp1(s)
    cuts = mysolution.minCut_dp2(s)
    m = len(dp)
    for x in xrange(m):
        if dp[x] != cuts[x]:
            print "x: " + str(x) + ", dp: " + str(dp[x]) + ", cuts: " + str(cuts[x])
    print "done"