

class Solution(object):
    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def dfs(k, s, address):
            # At the k-th segment, the length of s should be between 4-k and (4-k)*3
            if not ((4-k)<=len(s)<=(4-k)*3):
                return

            # At the last segment
            if k == 3:
                if ((1<len(s)<=3 and s[0]!='0') or len(s)==1) and 0<=int(s)<= 255:
                    res.append(address[1:] + '.' + s)
                return

            for l in xrange(1, 4):
                if l>1 and s[0]=='0':
                    return
                if 0 <= int(s[:l]) <= 255:
                    dfs(k + 1, s[l:], address + '.' + s[:l])
        res = []
        dfs(0,s,'')
        return res
