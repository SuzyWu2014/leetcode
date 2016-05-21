

class Solution(object):

    def longestPalindrome_DP1(self, s): # RT: O(n^2) TLE error, Space: O(n^2)
        n = len(s)
        dp = [[False]*n for _ in xrange(n)]

        # initialize the matrix
        for x in xrange(n):
            dp[x][x] = True
        for x in xrange(n-1):
            if s[x] == s[x+1]:
                dp[x][x+1] = True

        maxlength = 0
        idx = 0
        # populate the remaining cells of the matrix
        for length in xrange(3, n+1):
            for x in xrange(n-length+1):
                y = x+length-1
                if s[x] == s[y] and dp[x+1][y-1]:
                    dp[x][y] = True
                    if maxlength < length:
                        maxlength = length
                        idx = x
        return s[idx:idx+maxlength]

    def longestPalindrome_DP2(self, s):  # RT: O(n^2), Space: O(1)

        def expandAroundCenter(s, left, right):
            l, r = left, right
            n = len(s)
            while l >= 0 and r <= n-1 and s[l] == s[r]:
                l -= 1
                r += 1
            return s[l+1:r]

        n = len(s)
        if n == 0: return ""

        maxstring = s[0]
        for x in xrange(1, n):
            substring = expandAroundCenter(s, x-1, x)
            if len(maxstring) < len(substring):
                maxstring = substring

            substring = expandAroundCenter(s, x, x)
            if len(maxstring) < len(substring):
                maxstring = substring
        return maxstring
