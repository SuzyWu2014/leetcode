

class Solution(object):
    def myAtoi(self, str):
        """
        :type str: str
        :rtype: int
        """
        if len(str) == 0:
            return 0

        # remove the leading and tailing whitespace
        s = str.strip()

        # 0: positive, 1: negative
        neg = 0
        if s[0] == "+" or s[0] == "-":
            if s[0] == "-":
                neg = 1
            s = s[1:]

        lowerbound, upperbound = -2147483648, 2147483647
        res = 0
        for i in xrange(len(s)):
            if ord(s[i]) < 48 or ord(s[i]) > 57:
                return res

            res = res*10 + pow(-1, neg) * (ord(s[i])-48)

            if res < 0 and res <= lowerbound:
                return lowerbound
            elif res > 0 and res >= upperbound:
                return upperbound
        return res
