
class Solution:
    # @return a tuple, (index1, index2)
    def twoSum_dict(self, num, target):
        # use an dictionary to implement,
        # but there are no duplicates in num.
        # RT is O(n).
        dict = {}
        for i in range(len(num)):
            x = num[i]
            if target - x in dict:
                return dict[target-x], i
            dict[x] = i
        return -1, -1

    def twoSum_bruteforce(self, num, target):
        # brute force method: Use list to implement.
        # RT: O(n^2).
        if len(num) <= 1:
            return -1, -1

        list = []
        for i in range(len(num)):
            list.append((i, num[i]))

        list.sort(key=lambda x: x[1])

        for i in range(len(list)):
            a = list[i][1]
            b = target - a

            if i+1 < len(list):
                for j in xrange(i+1, len(list)):
                    if b == list[j][1]:
                        if list[i][0] <= list[j][0]:
                            return list[i][0], list[j][0]
                        else:
                            return list[j][0], list[i][0]
        return -1, -1

if __name__ == "__main__":
    mysolution = Solution()
    res = mysolution.twoSum_dict([0,4,3,0], 6)
    print res
