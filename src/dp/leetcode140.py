
class Solution(object):
    def wordBreak_dp1(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """
        # to find partition ways, need to use dfs based
        # on dp result
        Solution.result = []
        self.dfs(s, wordDict, '')
        return Solution.result

    def dfs(self, s, dict, stringlist):
        if self.check(s, dict):
            if len(s) == 0: Solution.result.append(stringlist[1:])
            for i in range(1, len(s) + 1):
                if s[:i] in dict:
                    self.dfs(s[i:], dict, stringlist + ' ' + s[:i])

    # check if s can be partitioned based on dict
    def check(self, s, dict):
        n = len(s)
        dp = [False for _ in xrange(n + 1)]
        dp[0] = True
        for l in xrange(1, n + 1):
            for i in range(l):
                if dp[i] and s[i:l] in dict:
                    dp[l] = True
                    break
        return dp[-1]



    def wordBreak_dp2(self, s, wordDict):
        """
        :type s: str
        :type wordDict: Set[str]
        :rtype: List[str]
        """
        self.wordlist = wordDict
        self.table = self.buildTable(s)
        self.res = []
        self.dfs(s, 0, '')
        return self.res

    def dfs(self, s, start, valuelist):
        """
        Depth-first traverse the decision tree by backtracking algorithm
        :param wordlist: the dictionary of words
        """
        # base case
        if start == len(s):
            self.res.append(valuelist[1:])

        # recursive step by backtracking
        for l in xrange(1, len(s)-start+1):
            if s[start:start+l] in self.wordlist and (start+l == len(s) or self.table[start+l][len(s)-1]):
                self.dfs(s, start+l, valuelist + ' ' + s[start:start+l])

    def buildTable(self, s):
        """
        check if s can be partitioned based on dict
        """
        n = len(s)
        # dp[i][j] denotes if s[i-1...j-1] can be segmented into the words in wordDict
        dp = [[False for j in xrange(n)] for i in xrange(n)]
        for l in xrange(1, n + 1):
            for i in xrange(n - l + 1):
                j = i + l - 1
                if s[i:j + 1] in self.wordlist:
                    dp[i][j] = True
                else:
                    for k in xrange(i + 1, j + 1):
                        if dp[i][k - 1] and dp[k][j]:
                            dp[i][j] = True
                            break
        return dp

if __name__ == "__main__":
    mysolution = Solution()
    res = mysolution.wordBreak_dp2("catsanddog", ["cat","cats","and","sand","dog"])
    print res