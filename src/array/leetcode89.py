
class Solution(object):
    def grayCode(self, n): # RT: O(n)
        res=[]
        size=1<<n
        for i in range(size):
            res.append((i>>1)^i)
        return res

if __name__ == "__main__":
    mysolution = Solution()
    res = mysolution.grayCode(3)
    print res
