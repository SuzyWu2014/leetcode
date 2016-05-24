
class Solution(object):
    def wordBreak_dp1(self, s, wordDict):   # O(n^2) time, O(n^2) space
        n = len(s)
        # dp[i][j] denotes if s[i...j] can be segmented into the words in wordDict
        dp = [[False for j in xrange(n)] for i in xrange(n)]
        for l in xrange(1, n+ 1):
            for i in xrange(n - l + 1):
                j = i + l - 1
                if s[i:j + 1] in wordDict:
                    dp[i][j] = True
                else:
                    for k in xrange(i + 1, j + 1):
                        if dp[i][k - 1] and dp[k][j]:
                            dp[i][j] = True
                            break
        return dp[0][n - 1]

    def wordBreak_dp2(self, s, wordDict):   # O(n^2) time, O(n) space
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: bool
        """
        n = len(s)
        # dp[i] indicates s[0..i-1] can be segmented into the words in wordDict
        dp = [False for _ in xrange(n + 1)]
        dp[0] = True
        for l in xrange(1, n + 1):
            for i in range(l):
                if dp[i] and s[i:l] in wordDict:
                    dp[l] = True
                    break
        return dp[-1]
