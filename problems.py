__author__ = 'Xin'

class ListNode(object):
    def __init__(self, x):
        self.val = x
        self.next = None

class RandomListNode(object):
    def __init__(self, x):
        self.label = x
        self.next = None
        self.random = None

class Interval(object):
    def __init__(self, s=0, e=0):
        self.start = s
        self.end = e

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, x):
        self.val = x
        self.left = None
        self.right = None

class Solution(object):

    def __init__(self):
        self.result = ""
        self.matrix = [["" for _ in range(1000)] for _ in range(1000)]

    def canCompleteCircuit(self, gas, cost):    # LTE, time is O(n^2)
        """
        :type gas: List[int]
        :type cost: List[int]
        :rtype: int
        """
        sum, total = 0, 0
        start = -1
        for i in range(len(gas)):
            sum += gas[i]-cost[i]
            total += gas[i]-cost[i]
            if sum<0:
                sum = 0
                start = i

        if total<0:
            return -1
        else:
            return start+1

    def candy(self, ratings):
        """
        :type ratings: List[int]
        :rtype: int
        """
        n = len(ratings)
        candynum = [1 for _ in range(n)]

        for i in range(1, n):
            if ratings[i] > ratings[i-1]:
                candynum[i] = candynum[i-1] + 1

        for i in reversed(range(n-1)):
            if ratings[i] > ratings[i+1] and candynum[i] <= candynum[i+1]:
                candynum[i] = candynum[i+1] + 1

        return sum(candynum)

    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        first = 0
        last = len(nums)
        while first!=last:
            mid = (first + last)/2

            if nums[mid] == target:
                return mid

            if nums[first]<=nums[mid]:
                if nums[first]<=target and target<nums[mid]:
                    last = mid
                else:
                    first = mid+1
            else:
                if nums[mid]<target and target<=nums[last-1]:
                    first = mid+1
                else:
                    last = mid
        return -1

    def searchII(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: bool
        """
        first = 0
        last = len(nums)
        while first != last:
            mid = (first+last)/2

            if nums[mid] == target:
                return True

            if nums[first]<nums[mid]:
                if nums[first]<=target and target<nums[mid]:
                    last = mid
                else:
                    first = mid + 1
            elif nums[first]>nums[mid]:
                if nums[mid]<target and target<=nums[last-1]:
                    first = mid + 1
                else:
                    last = mid
            else:
                first += 1

        return False

    def threeSum(self, nums):
        if len(nums)<3:
            return []
        nums.sort() # O(nlogn)
        result = []

        for i in range(len(nums)-2):
            if i==0 or nums[i]>nums[i-1]:
                left = i+1
                right = len(nums)-1

                while left < right:
                    val = nums[left] + nums[right]
                    if val == -nums[i]:
                        result.append([nums[i], nums[left], nums[right]])
                        left += 1
                        right -= 1
                        while left < right and nums[left]==nums[left-1]:
                            left += 1
                        while left < right and nums[right]==nums[right+1]:
                            right -= 1
                    elif val < -nums[i]:
                        while left < right:
                            left += 1
                            if nums[left] > nums[left-1]:
                                break
                    else:
                        while left < right:
                            right -= 1
                            if nums[right] < nums[right+1]:
                                break
        return result

    def threeSum2(self, nums):   # causes TLE
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """
        nums.sort() # O(nlog(n))

        # This is a generic k-sum algorithm
        def ksum(nums,k,target):
            result = []
            i=0

            if k==2:
                j = len(nums)-1
                while i<j:
                    if nums[i]+nums[j]==target:
                        result.append([nums[i],nums[j]])
                        i+=1
                    elif nums[i]+nums[j]>target:
                        j-=1
                    else:
                        i+=1
            else:   # case: k>2
                while i<len(nums)-k+1:
                    newtarget = target - nums[i]
                    subresult = ksum(nums[i+1:], k-1, newtarget)
                    if subresult:
                        result = result + [[nums[i],]+nr for nr in subresult]
                    i+=1
            return result
        return ksum(nums, 3, 0)

    def addTwoNumbers(self, l1, l2):
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 == None:
            return l2
        if l2 == None:
            return l1

        h1 = l1
        h2 = l2
        while h1!=None and h2!=None:
            h1 = h1.next
            h2 = h2.next

        if h2!=None:
            l1,l2 = l2,l1

        head = l1
        carry = 0
        while l2!=None:
            val = l1.val + l2.val + carry
            l1.val = val%10
            carry = val/10
            tail = l1
            l1 = l1.next
            l2 = l2.next

        if carry==0:
            return head
        else:
            while l1!=None:
                val = l1.val + carry
                l1.val = val%10
                carry = val/10
                if carry==0:
                    break
                tail = l1
                l1 = l1.next
            if carry==0:
                return head
            else:
                tail.next = ListNode(carry)
                return head

    def reverseBetween(self, head, m, n):
        """
        :type head: ListNode
        :type m: int
        :type n: int
        :rtype: ListNode
        """
        if head == None or head.next == None:
            return head

        dummy = ListNode(0)
        dummy.next = head
        head1 = dummy

        for i in range(m-1):
            head1 = head1.next
        p = head1.next # points to the m-th node
        curr = p
        next = curr.next
        for i in range(n-m):
            tmp = curr
            curr = next
            next = curr.next
            curr.next = tmp
        head1.next = curr
        p.next = next
        return dummy.next

    def partition(self, head, x):
        """
        :type head: ListNode
        :type x: int
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return head

        curr = head

        small_head = ListNode(0)
        small_tail = small_head

        large_head = ListNode(0)
        large_tail = large_head

        while curr != None:
            while curr != None and curr.val < x:
                small_tail.next = curr
                small_tail = small_tail.next
                curr = curr.next
            small_tail.next = None

            while curr != None and curr.val >= x:
                large_tail.next = curr
                large_tail = large_tail.next
                curr = curr.next
            large_tail.next = None

        small_tail.next = large_head.next
        return small_head.next

    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return head
        curr = head
        while curr!=None:
            while curr.next!=None and curr.val!=curr.next.val:
                curr = curr.next
            if curr.next==None:
                return head
            p = curr.next.next
            while p!=None and p.val==curr.val:
                p = p.next
            if p==None:
                curr.next = None
                return head
            else:
                curr.next=p
                curr = curr.next
        return head

    def deleteDuplicatesII(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return head
        dummy = ListNode(0)
        pre = dummy
        dummy.next = head
        curr = head
        while curr!=None and curr.next!=None:
            if curr.val == curr.next.val:
                while curr.next!=None and curr.val == curr.next.val:
                    curr = curr.next
                curr = curr.next
                pre.next = curr
            else:
                pre = curr
                curr = curr.next
        return dummy.next

    def removeNthFromEnd(self, head, n):
        """
        :type head: ListNode
        :type n: int
        :rtype: ListNode
        """
        if head==None or (head.next==None and n==1):
            return None

        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        count = 0 # the number of nodes
        while p.next!=None:
            p = p.next
            count += 1
        p = dummy
        step = count - n
        for i in range(step):
            p = p.next

        p.next = p.next.next
        return dummy.next

    def swapPairs(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return head

        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        c = dummy.next
        while c!=None and c.next!=None:
            tmp = c.next
            c.next = c.next.next
            tmp.next = c
            p.next = tmp

            p = c
            c = c.next

        return dummy.next

    def reverseKGroup(self, head, k):
        """
        :type head: ListNode
        :type k: int
        :rtype: ListNode
        """
        if head==None or k==1:
            return head

        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        c = dummy.next
        while c!=None:
            for i in range(k-1):
                c = c.next
                if c==None:
                    return dummy.next
            head1 = p.next
            tail1 = c
            c = c.next
            tail1.next = None

            # reverse the part from head1 to tail1
            pre = head1
            curr = pre.next
            pre.next = None
            while curr!=None:
                next = curr.next
                curr.next = pre
                pre = curr
                curr = next

            head1.next = c
            p.next = tail1
            p = head1

        return dummy.next

    def copyRandomList(self, head):
        """
        :type head: RandomListNode
        :rtype: RandomListNode
        """
        if head==None:
            return head

        p = head
        while p!=None:
            node = RandomListNode(p.label)
            tmp = p.next
            node.next = tmp
            p.next = node
            p = node.next

        p = head
        while p!=None:
            if p.random!=None:
                p.next.random = p.random.next
            p = p.next.next

        p = head.next
        while p.next!=None:
            p.next = p.next.next
            p = p.next

        return head.next

    def reorderList(self, head):
        """
        :type head: ListNode
        :rtype: void Do not return anything, modify head in-place instead.
        """
        if head==None or head.next==None or head.next.next==None:
            return head

        p = head.next.next
        q = head.next
        while p!=None:
            if p.next!=None and p.next.next!=None:
                p = p.next.next
                q = q.next
            elif p.next!=None: # even
                p = p.next
                tmp = q
                q = q.next
                tmp.next = None
                break
            else: # odd
                tmp = q
                q = q.next
                tmp.next = None
                break

        # reverse from q to p
        if q!=p:
            pre = q
            q = q.next
            pre.next = None
            while q!=None:
                tmp = q.next
                q.next = pre
                pre = q
                q = tmp

        q = head
        while p!=None :
            tmp = p.next
            p.next = q.next
            q.next = p
            q = p.next
            p = tmp

        return head

    def longestPalindrome(self, s):
        """
        :type s: str
        :rtype: str
        """
        if len(s)<2:
            return s

        n = len(s)
        i = 0
        while i<n:
            if s[i]!=s[n-1-i]:
                break
            i+=1
        if i==n:
            return s

        for i in range(1,n-1):
            if len(self.result) > n-i:
                break

            if len(self.matrix[i][n-1]) > 0:
                lp1 = self.matrix[i][n-1]
            else:
                lp1 = self.longestPalindrome(s[i:])
            if len(lp1)>len(self.result):
                self.result = lp1

            if len(self.matrix[0][n-i-1]) > 0:
                lp2 = self.matrix[0][n-i-1]
            else:
                lp2 = self.longestPalindrome(s[:n-i])
            if len(lp2)>len(self.result):
                self.result = lp2

        return self.result

    def isMatch(self, s, p): # RT: TLE error
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        # memorization matrix: dp[s_i][p_j], the indices start from 1.
        dp=[[False for i in range(len(p)+1)] for j in range(len(s)+1)]
        dp[0][0]=True
        for i in range(1,len(p)+1):
            if p[i-1]=='*':
                if i>=2:
                    dp[0][i]=dp[0][i-2]
        for i in range(1,len(s)+1):
            for j in range(1,len(p)+1):
                if p[j-1]=='.':
                    dp[i][j]=dp[i-1][j-1]
                elif p[j-1]=='*':
                    dp[i][j]=dp[i][j-1] or dp[i][j-2] or (dp[i-1][j] and (s[i-1]==p[j-2] or p[j-2]=='.'))
                else:
                    dp[i][j]=dp[i-1][j-1] and s[i-1]==p[j-1]
        return dp[len(s)][len(p)]

    def isMatch_recurrsive(self, s, p): # RT: TLE error
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if len(p)==0:
            return len(s)==0

        if len(p)==1 or p[1]!="*":
            if len(s)==0:
                return False
            elif s[0]!=p[0] and p[0]!=".":
                return False
            else:
                return self.isMatch(s[1:], p[1:])
        else: # len(p)>1 and p[1]=="*"
            i = -1
            while i<len(s) and (i<0 or p[0]=="." or p[0]==s[i]):
                if self.isMatch_recurrsive(s[i+1:], p[2:]): # case: p=".*xxxx"
                    return True
                i += 1
            return False

    def longestValidParentheses1(self, s):
        """
        :type s: str
        :rtype: int
        """
        if len(s)<2:
            return 0

        maxlen = 0
        pairs = 0
        stack = []
        i = 0
        while i<len(s) and s[i]==")":
            i+=1
        str = s[i:]
        for i in range(len(str)):
            if str[i]=="(":
                stack.append(str[i])
            if str[i]==")":
                if len(stack)>0:
                    stack.pop()
                    pairs += 1

                    if i==len(str)-1:
                        maxlen = max(maxlen, pairs)
                else:
                    maxlen = max(maxlen, pairs)
                    pairs = 0
        return 2*maxlen

    def longestValidParentheses(self, s):
        """
        :type s: str
        :rtype: int
        """
        maxlen = 0
        stack = []
        last = -1
        for i in range(len(s)):
            if s[i]=='(':
                stack.append(i)     # push the INDEX into the stack!!!!
            else:
                if stack == []:
                    last = i
                else:
                    stack.pop()
                    if stack == []:
                        maxlen = max(maxlen, i-last)
                    else:
                        maxlen = max(maxlen, i-stack[len(stack)-1])
        return maxlen

    def largestRectangleArea(self, heights):
        """
        :type heights: List[int]
        :rtype: int
        """
        if len(heights) == 0:
            return 0

        height_stack = []
        index_stack = []
        maxarea = heights[0]

        for i in range(0, len(heights)):

            if height_stack==[] or (heights[i] > height_stack[len(height_stack)-1]):
                height_stack.append(heights[i])
                index_stack.append(i)
            elif heights[i] < height_stack[len(height_stack)-1]:
                lastIndex = 0
                while height_stack and heights[i] < height_stack[len(height_stack)-1]:
                    height = height_stack.pop()
                    lastIndex = index_stack.pop()
                    area = height * (i - lastIndex)
                    maxarea = max(maxarea, area)

                height_stack.append(heights[i])
                index_stack.append(lastIndex)

        while height_stack:
            height = height_stack.pop()
            index = index_stack.pop()
            area = height * (len(heights) - index)
            maxarea = max(maxarea, area)

        return maxarea

    def evalRPN(self, tokens):
        """
        :type tokens: List[str]
        :rtype: int
        """
        stack = []
        for t in tokens:
            if t!="+" and t!="-" and t!="*" and t!="/":
                stack.append(int(t))
            else:
                a = stack.pop()
                b = stack.pop()

                if t=="+":
                    stack.append(a + b)
                elif t=="-":
                    stack.append(b - a)
                elif t=="*":
                    stack.append(a * b)
                else:
                    if a<0 and b>0 and abs(a)>b:
                        stack.append(b/abs(a) * (-1))
                    else:
                        stack.append(b/a)
        return stack.pop()

    def romanToInt(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        dict = {'M':1000,'D':500,'C':100,'L':50,'X':10,'V':5,'I':1}
        result = dict[s[n-1]]
        i = n-2
        while i>=0:
            if dict[s[i]] >= dict[s[i+1]]:
                result += dict[s[i]]
            else:
                result -= dict[s[i]]
            i -= 1
        return result

    def sortList(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        if head==None or head.next==None:
            return head

        fast = slow = head
        while fast.next and fast.next.next:
            fast = fast.next.next
            slow = slow.next
        head1 = head
        head2 = slow.next
        slow.next = None
        head1 = self.sortList(head1)
        head2 = self.sortList(head2)
        head = self.merge(head1, head2)
        return head

    def merge(self, head1, head2):
        if head1 == None: return head2
        if head2 == None: return head1
        dummy = ListNode(0)
        p = dummy
        while head1 and head2:
            if head1.val <= head2.val:
                p.next = head1
                head1 = head1.next
                p = p.next
            else:
                p.next = head2
                head2 = head2.next
                p = p.next
        if head1:
            p.next = head1
        if head2:
            p.next = head2
        return dummy.next

    def minimumTotal(self, triangle):
        if len(triangle) == 0: return 0
        array = [0 for i in range(len(triangle))]
        array[0] = triangle[0][0]
        for i in range(1, len(triangle)):
            for j in range(len(triangle[i]) - 1, -1, -1):
                if j == len(triangle[i]) - 1:
                    array[j] = array[j-1] + triangle[i][j]
                elif j == 0:
                    array[j] = array[j] + triangle[i][j]
                else:
                    array[j] = min(array[j-1], array[j]) + triangle[i][j]
        return min(array)

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums==None or len(nums)==0:
            return 0
        if len(nums)==1:
            return nums[0]

        import sys
        maxsum = -sys.maxint-1

        n = len(nums)
        # n*n matrix as a memorization
        memo = [[0 for _ in range(n)] for _ in range(n)]
        memo[0][0] = nums[0]
        for i in range(len(nums)):
            for j in range(i,-1,-1):
                if j==i:
                    memo[i][j] = nums[i]
                else:
                    memo[i][j] = memo[i][i] + memo[i-1][j]
                maxsum = max(maxsum, memo[i][i])
        return maxsum

    def minCut_old(self, s):
        """
        :type s: str
        :rtype: int
        """
        # the min cuts of the string s[i:]
        dp = [0 for i in range(len(s)+1)]
        # p[i][j] is True, if s[i,j] is palindrome; otherwise, False
        p = [[False for i in range(len(s))] for j in range(len(s))]
        # set dp[i] the upper bound
        for i in range(len(s)+1):
            dp[i] = len(s) - i
        for i in range(len(s)-1, -1, -1):
            for j in range(i, len(s)):
                if s[i] == s[j] and (((j - i) < 2) or p[i+1][j-1]):
                    p[i][j] = True
                    dp[i] = min(1+dp[j+1], dp[i])
        return dp[0]-1

    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)
        dp = [[-1 for _ in range(n)] for _ in range(n)]
        for l in range(1, n+1):
            for i in range(n-l+1):
                j = i+l-1
                if i == j:
                    dp[i][j] = 0
                elif self.isPalindrome(s[i:j+1]):
                    dp[i][j] = 0
                else:
                    for k in range(i, j):
                        # dp[i][j] = 1 + min(dp[i][k], dp[k+1][j])
                        tmp = 1 + dp[i][k] + dp[k+1][j]
                        if dp[i][j] < 0:
                            dp[i][j] = tmp
                        else:
                            dp[i][j] = min(dp[i][j], tmp)
        return dp[0][n-1]

    def isPalindrome(self, s):
        if len(s)>=2:
            for i in range(len(s)):
                if s[i] != s[len(s)-1-i]:
                    return False
        return True

    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if len(matrix)==0:
            return 0
        a = [0 for _ in range(len(matrix[0]))]
        maxarea = 0
        for i in range(len(matrix)):
            for j in range(len(matrix[i])):
                a[j] = a[j]+1 if matrix[i][j] == 1 else 0

            maxarea = max(maxarea, self.largestRectangle(a))
        return maxarea

    def largestRectangle(self, height):
        stack=[]; area = 0
        i = 0
        while i<len(height):
            if stack==[] or height[i]>height[stack[-1]]:
                stack.append(i)
                i += 1
            else:
                curr = stack.pop()
                width = i if stack==[] else i-stack[-1]-1
                area = max(area, width * height[curr])

        while stack!=[]:
            curr = stack.pop()
            width = i if stack==[] else len(height)-stack[-1]-1
            area = max(area, width * height[curr])

        return area

    def numDistinct(self, s, t):
        """
        :type s: str
        :type t: str
        :rtype: int
        """
        dp = [[0 for _ in range(len(t)+1)] for _ in range(len(s)+1)]

        for i in range(len(s)+1):
            dp[i][0] = 1

        for i in range(1, len(s)+1):
            for j in range(1, min(i+1, len(t)+1)):
                if s[i-1] == t[j-1]:
                    dp[i][j] = dp[i-1][j] + dp[i-1][j-1]
                else:
                    dp[i][j] = dp[i-1][j]

        return dp[len(s)][len(t)]

    def numScore(self, score):
        dp = [0,0]
        for i in range(2, score+1):
            if i==2 or i==3:
                dp.append(1)
            else:
                ways = dp[i-2] + dp[i-3] if 3<i<=7 else dp[i-2] + dp[i-3] + dp[i-7]
                if i==7:
                    ways += 1
                dp.append(ways)
        return dp[score]

    def minDistance(self, word1, word2):
        """
        :type word1: str
        :type word2: str
        :rtype: int
        """
        m = len(word1)+1; n = len(word2)+1
        dp = [[0 for _ in range(n)] for _ in range(m)]
        for i in range(n):
            dp[0][i] = i
        for i in range(m):
            dp[i][0] = i
        for i in range(1, m):
            for j in range(1, n):
                dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(0 if word1[i-1]==word2[j-1] else 1))

        return dp[m-1][n-1]

    def isMatch_str(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        star = -1 # record the position of * in p
        ppointer = spointer = ss = 0
        while spointer < len(s):
            if ppointer<len(p) and (s[spointer]==p[ppointer] or p[ppointer]=='?'):
                spointer += 1; ppointer += 1
                continue
            if ppointer<len(p) and p[ppointer]=='*':
                star = ppointer; ss = spointer; ppointer+=1
                continue
            if star!=-1:
                ppointer = star+1; ss += 1; spointer = ss
                continue
            return False
        while ppointer<len(p) and p[ppointer]=='*':
            ppointer += 1
        if ppointer == len(p):
            return True
        return False

    def isMatch_rec(self, s, p):
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if len(p)>len(s):
            return False
        elif len(s)==0 or len(p)==0:
            return len(s) == len(p)
        elif len(s)==1 and len(p)==1 and s==p:
            return True
        elif s[0]==p[0] or p[0]=='?':
            return self.isMatch_rec(s[1:],p[1:])
        elif p[0]=='*':
            i = 0
            while i<len(p) and p[i]=='*':
                i += 1
            if i==len(p): return True
            j = 0
            while j<len(p) and not self.isMatch_rec(s[j:], p[i:]):
                j += 1
            return j!=len(p)
        return False

    def isMatch_dp(self, s, p): # RT: TLE error
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        m, n = len(s), len(p)
        if n==0: return m==0
        if m==0: return p=="*" or n==0
        if n - p.count('*') > m: return False

        q = []
        isFirst = True
        for i in range(n):
            if p[i]=='*':
                if isFirst:
                    q.append(p[i])
                    isFirst = False
            else:
                q.append(p[i])
                isFirst = True
        p = q
        n = len(p)

        # initialize dp matrix
        dp=[[False for _ in range(n+1)] for _ in range(m+1)]
        dp[0][0]=True
        if p[0]=='*':
            for i in range(0, m+1):
                dp[i][1] = True

        for i in range(1, m+1): # for string
            for j in range(1, n+1): # for pattern
                if p[j-1]=='*':
                    dp[i][j] = dp[i-1][j] or dp[i][j-1]
                elif s[i-1]==p[j-1] or p[j-1]=='?':
                    dp[i][j] = dp[i-1][j-1]
                else:
                    dp[i][j] = False
        return dp[m][n]

    _dp = [0]
    def numSquares(self, n):
        dp = self._dp
        while len(dp) <= n:
            dp += min(dp[-i*i] for i in range(1, int(len(dp)**0.5+1))) + 1,
        return dp[n]

    def minCut_ddpp(self, s): # TLE error
        """
        :type s: str
        :rtype: int
        """
        # the min cuts of the string s[i:]
        dp = [0 for i in range(len(s)+1)]
        # p[i][j] is True, if s[i,j] is palindrome; otherwise, False
        p = [[False for i in range(len(s))] for j in range(len(s))]
        # set dp[i] the upper bound
        for i in range(len(s)+1):
            dp[i] = len(s) - i
        for i in range(len(s)-1, -1, -1):
            for j in range(i, len(s)):
                # case1: i==j
                # case2: i==j+1, and s[i]==s[j]
                # case3: s[i]==s[j], and s[i+1...j-1] is palindrome
                if s[i] == s[j] and (((j - i) < 2) or p[i+1][j-1]):
                    p[i][j] = True
                    dp[i] = min(1+dp[j+1], dp[i])
        return dp[0]-1

    def maxProduct(self, nums): # RT: O(n)
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n==0: return n
        maxval = max(nums)
        maxp = [0 for _ in range(n)] # save the current max val
        minp = [0 for _ in range(n)] # save the current min val
        if nums[0] > 0: maxp = nums[0]
        elif nums[0] < 0: minp = nums[0]

        for i in range(1, n):
            if nums[i] > 0:
                maxp[i] = max(maxp[i-1]*nums[i], nums[i])
                minp[i] = minp[i-1] * nums[i]
            elif nums[i] < 0:
                minp[i] = min(maxp[i-1]*nums[i], nums[i])
                maxp[i] = minp[i-1]*nums[i]
            maxval = max(maxp[i], maxval)
        return maxval

    def lengthOfLIS(self, nums): # RT: O(nlogn)
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n==0: return n
        dp = [0 for _ in range(n)]
        length = 0
        for i in range(1, n):
            if nums[dp[0]] > nums[i]:
                dp[0] = i
            elif nums[dp[length]] < nums[i]:
                length += 1
                dp[length] = i
            else:
                # do a binary search to find the ceiling
                # of nums[i] and put it there
                index = self.ceilIndex(nums, dp, length, nums[i])
                dp[index] = i
        return length

    def ceilIndex(self, nums, dp, length, val):
        start = 0; end = length
        while start <= end:
            mid = (start+end)/2
            if mid<length and nums[dp[mid]]<val<=nums[dp[mid+1]]:
                return mid+1
            elif val < nums[dp[mid]]:
                end = mid - 1
            else:
                start = mid + 1
        return -1

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n==0: return 0
        dp = [0 for _ in range(n+1)]
        minprice = prices[0]
        for x in range(2,n+1):
            if dp[x-2]>dp[x-1] and dp[x-2]>(prices[x-1]-minprice):
                dp[x] = dp[x-2]
                minprice = prices[x-1]
            else:
                #dp[x] = max(prices[x-1]-minprice, dp[x-1])
                if prices[x-1]-minprice > dp[x-1]:
                    dp[x] = prices[x-1]-minprice
                    if x < n-2:
                        minprice = prices[x+1]
                else:
                    dp[x] = dp[x-1]
                    if x < n-1:
                        minprice = prices[x]
        return dp[n]

    def partition_dp(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        n = len(s)
        if n==0 or n==1: return [[s]]
        dp = [[False]*n for _ in range(n)]
        for i in range(n-1,-1,-1):
            for j in range(i,n):
                dp[i][j] = s[i]==s[j] and ((j-i)<2 or dp[i+1][j-1]==True)

        res = [[] for _ in range(n)]
        for i in range(n-1,-1,-1):
            for j in range(i,n):
                if dp[i][j]==True:
                    palindrome = s[i:j+1]
                    if j+1<n:
                        tmp = []
                        for list in res[j+1]:
                            tmp.append(palindrome)
                            tmp += list
                            res[i].append(tmp)
                    else:
                        res[i].append([palindrome])

        return res[0]

    def solveNQueens(self, n):
        """
        :type n: int
        :rtype: List[List[str]]
        """
        def check(k,j): # check if the k-th Queen can be put in column j
            for i in range(k):
                # the first condition checks if the j-th column is available
                # the second condition checks if the diagonal lines are available
                if board[i]==j or abs(k-i)==abs(board[i]-j):
                    return False
            return True

        def dfs(depth, valuelist):
            if depth==n: res.append(valuelist)
            else:
                for i in range(n):
                    if check(depth, i):
                        board[depth] = i
                        s = '.' * n
                        dfs(depth+1, valuelist+[s[:i]+'Q'+s[i+1:]])

        board = [-1 for _ in range(n)]
        res = []
        dfs(0, [])
        return res

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def dfs(s, sub, ips, ip):
            if sub==4:
                if s=='':
                    ips.append(ip[1:])
                return

            for i in range(1, 4):
                if i <= len(s):
                    if int(s[:i])<=255:
                        dfs(s[i:], sub+1, ips, ip+'.'+s[:i])
                    if s[0]=='0': break

        ips = []
        dfs(s, 0, ips, '')
        return ips

    def combinationSum(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(nums, t, valuelist):
            if t==0:
                tmp = sorted(valuelist)
                if tmp not in res:
                    res.append(tmp)
                return
            else:
                for num in nums:
                    if num<=t:
                        newsum = t-num
                        dfs(nums, newsum, valuelist+[num])
        candidates.sort()
        res = []
        dfs(candidates, target, [])
        return res

    def combinationSum2(self, candidates, target):
        """
        :type candidates: List[int]
        :type target: int
        :rtype: List[List[int]]
        """
        def dfs(nums, t, start, valuelist):
            if t==0:
                return res.append(valuelist)
            else:
                for i in range(start, len(nums)):
                    if t<nums[i]: return
                    dfs(nums, t-nums[i], i+1, valuelist+[nums[i]])

        candidates.sort()
        res = []
        dfs(candidates, target, 0, [])
        return res

    def searchInsert(self, A, target):
        left = 0; right = len(A) - 1
        while left <= right:
            mid = ( left + right ) / 2
            if A[mid] < target:
                left = mid + 1
            elif A[mid] > target:
                right = mid - 1
            else:
                return mid
        return left

    def spiralOrder(self, matrix):
        """
        :type matrix: List[List[int]]
        :rtype: List[int]
        """
        res = []
        if matrix == []: return res
        m, n = len(matrix), len(matrix[0])
        # if m==1:
        #     for y in range(0,n):
        #         res.append(matrix[0][y])
        # elif n==1:
        #     for x in range(0,m):
        #         res.append(matrix[x][0])
        # else:
        d = 0
        while d <= (min(m,n)-1)/2:
            x = d
            for y in range(d, n-d):
                res.append(matrix[x][y])
            y = n-1-d
            for x in range(d+1, m-d):
                res.append(matrix[x][y])
            if m-2*d>1 and n-2*d>1:
                x = m-1-d
                for y in range(n-2-d, d-1, -1):
                    res.append(matrix[x][y])
                y = d
                for x in range(m-2-d, d, -1):
                    res.append(matrix[x][y])
            d += 1
        return res

    def merge(self, intervals):
        """
        :type intervals: List[Interval]
        :rtype: List[Interval]
        """
        intervals.sort()
        m = len(intervals)
        res = []
        tmp = intervals[0]
        for x in range(1,m):
            if tmp.start<=intervals[x].start<=tmp.end<=intervals[x].end:
                tmp.end = intervals[x].end
            elif tmp.end<intervals[x].start:
                res.append(tmp)
                tmp = intervals[x]
        res.append(tmp)
        return res

    def canJump(self, nums):
        """
        :type nums: List[int]
        :rtype: bool
        """
        n = len(nums)
        x = 0
        while (x<n-1) or (n==1):
        #for x in range(n-1):
            size = n-x
            sum = 0
            while x<n-1 and sum<n-1:
                if nums[x]==0: break
                sum += nums[x]
                x += nums[x]
            if sum>=size-1:
                return True
            if x==0:
                return False
            x+=1
        return False

    def longestConsecutive(self, nums):
        n = len(nums)
        if n <= 1: return n
        dict = {}
        for x in nums:
            dict[x] = False
        maxlength = 0
        for x in nums:
            if dict[x]: continue
            length = 1
            dict[x] = True

            y = x+1
            while dict.has_key(y):
                dict[y] = True
                length += 1
                y += 1

            y = x-1
            while dict.has_key(y):
                dict[y] = True
                length += 1
                y -= 1

            maxlength = max(maxlength, length)
            if maxlength == n: break
        return maxlength

    def twoSum(self, nums, target):
        dict = {}
        for x in xrange(len(nums)):
            dict[nums[x]] = x+1

        for x in nums:
            k = target - x
            if k>0 and dict.has_key(k) and dict[k]!=dict[x]:
                return (dict[x], dict[k])

        return (-1,-1)

    def threeSum(self, nums, target):
        nums.sort()
        return [list(t) for t in self.ksum(nums, 3, target)]

    def ksum(self, nums, k, target):
        n = len(nums)
        res = set()
        i = 0
        if k==2:
            j = n-1
            while i < j:
                if nums[i]+nums[j] > target:
                    j -= 1
                elif nums[i]+nums[j] < target:
                    i += 1
                else:
                    res.add((nums[i], nums[j]))
                    i += 1
        else:
            j = n-k+1
            while i < j:
                newtarget = target - nums[i]
                newres = self.ksum(nums[i+1:], k-1, newtarget)
                if newres:
                    res |= set((nums[i],) + r for r in newres)
                i += 1
        return res

    def threeSumClosest(self, nums, target):
        nums.sort()
        import sys
        min_gap = sys.maxint
        result = 0

        for i in range(len(nums)-3+1):
            j = i+1
            k = len(nums)-1
            while j < k:
                sum = nums[i] + nums[j] + nums[k]
                gap = abs(target - sum)
                if gap < min_gap:
                    min_gap = gap
                    result = sum
                    if min_gap == 0:
                        return target
                if sum < target:
                    j += 1
                else:
                    k -= 1
        return result

    def fourSum(self, nums, target):
        nums.sort()
        return [list(t) for t in self.mimic_ksum(nums, 4, target)]

    def mimic_ksum(self, nums, k, target):
        n = len(nums)
        res = set()
        i = 0
        if k == 2:
            j = n-1
            while i < j:
                if target > nums[i]+nums[j]:
                    i += 1
                elif target < nums[i]+nums[j]:
                    j -= 1
                else:
                    res.add((nums[i], nums[j]))
                    i += 1
        else:
            j = n-k+1
            while i < j:
                newtarget = target - nums[i]
                newres = self.mimic_ksum(nums[i+1:], k-1, newtarget)
                if newres:
                    res |= set((nums[i], ) + nr for nr in newres)
                i += 1
        return res

    def removeElement(self, nums, val):
        idx = 0
        for item in nums:
            if item != val:
                nums[idx] = item
                idx += 1
        return idx

    def nextPermutation(self, nums):
        n = len(nums)
        idxPartitionNumber = -1
        for x in xrange(n-1, -1, -1):
            if nums[x-1] < nums[x]:
                idxPartitionNumber = x-1
                break
        if idxPartitionNumber < 0:
            nums.reverse()
        else:
            for x in xrange(n-1, -1, -1):
                if nums[x] > nums[idxPartitionNumber]:
                    nums[x], nums[idxPartitionNumber] = nums[idxPartitionNumber], nums[x]
                    break
            nums[idxPartitionNumber+1:] = nums[idxPartitionNumber+1:][::-1]
        print nums

    def fourSum(self, nums, target):
        nums.sort()
        n = len(nums)
        res = set()
        dict = {}

        # establish the dict
        for i in xrange(n-1):
            for j in xrange(i+1, n):
                sum = nums[i]+nums[j]
                # there could be duplicates
                if dict.has_key(sum):
                    dict[sum].append((i,j))
                else:
                    dict[sum] = [(i,j)]

        for i in xrange(n-4+1):
            for j in xrange(i+1, n-4+2):
                newsum = target - nums[i] - nums[j]
                if dict.has_key(newsum):
                    for item in dict[newsum]:
                        # Elements in a quadruplet (a,b,c,d)
                        # must be in non-descending order
                        if j < item[0]:
                            res.add((nums[i],nums[j],nums[item[0]],nums[item[1]]))

        return [list(t) for t in res]

    def getPermutation(self, n, k):
        nums = [x+1 for x in xrange(n)]
        return self.helper(nums, n, k)

    def helper(self, nums, n, k):
        if n==1:
            return nums[0]
        import math
        idx = (k-1)/math.factorial(n-1)
        return str(nums[idx]) + self.helper(nums[:idx]+nums[idx+1:], n-1, (k-1) % math.factorial(n-1))

    def getPermutation2(self, n, k):
        res = ''
        # compute the factorial of (n-1)
        factorial = 1
        for i in xrange(1,n): factorial *= i
        k -= 1
        nums = [x+1 for x in xrange(n)]
        for x in xrange(n-1, -1, -1):
            i = k/factorial
            res += str(nums[i])
            nums.remove(nums[i])
            if x != 0:
                k = k % factorial
                factorial = factorial / x
        return res

    def trap(self, height):
        n = len(height)
        # from left to right, compute leftmost[i],
        # which means the most height on the left
        # side of the i-th position
        leftmax = 0
        leftmost = []
        for x in xrange(n):
            if height[x] > leftmax:
                leftmax = height[x]
            leftmost.append(leftmax)

        water = 0
        # rightmost indicates the most height on the
        # right side of the i-th position from right
        # to left
        rightmost = 0
        for x in xrange(n-1, -1, -1):
            if height[x] > rightmost:
                rightmost = height[x]
            if min(leftmost[x], rightmost) > height[x]:
                water += min(leftmost[x], rightmost)-height[x]

        return water

    def rotateimage(self, matrix):
        n = len(matrix)
        for x in xrange(n/2):
            matrix[x], matrix[n-1-x] = matrix[n-1-x], matrix[x]

        k = 0 # indicates the starting column
        for x in xrange(n):
            for y in xrange(k, n):
                if x != y:
                    matrix[x][y], matrix[y][x] = matrix[y][x], matrix[x][y]
            k += 1
        print matrix

    def plusOne(self, digits):
        n = len(digits)
        carry = 1
        for x in xrange(n-1,-1,-1):
            val = digits[x] + carry
            carry = val / 10
            digits[x] = val % 10
        if carry == 1:
            digits = [1] + digits
        return digits

    def climbStairs_recurrsive(self, n):
        if n == 1: return 1
        if n == 2: return 2
        return self.climbStairs_recurrsive(n-1) + self.climbStairs_recurrsive(n-2)

    def climbStairs_dp(self, n):
        dp = [0 for _ in xrange(n+1)]
        dp[0] = 1
        dp[1] = 1
        for x in xrange(2, n+1):
            dp[x] = dp[x-1] + dp[x-2]
        return dp[n]

    def grayCode(self, n):
        res=[]
        size=1<<n
        for i in range(size):
            res.append((i>>1)^i)
        return res

    def setzeroes(self, matrix):
        m,n = len(matrix), len(matrix[0])
        zerorows = [False for _ in xrange(m)]
        zerocols = [False for _ in xrange(n)]

        for x in xrange(m):
            for y in xrange(n):
                if matrix[x][y] == 0:
                    zerorows[x], zerocols[y] = True, True

        for x in xrange(m):
            if zerorows[x] == True:
                matrix[x] = [0 for _ in xrange(n)]

        for y in xrange(n):
            if zerocols[y] == True:
                for x in xrange(m):
                    matrix[x][y] = 0

        print matrix

    def findLadders(self, beginWord, endWord, wordlist):
        n = len(beginWord)
        previous = set()
        current = set()
        current.add(beginWord)
        previousMap = {}
        for word in wordlist:
            previousMap[word] = []
        res = []

        while True:
            previous, current = current, previous
            for word in current:
                wordlist.remove(word)
            current.clear()
            for word in previous:
                for i in xrange(n):
                    part1, part2 = word[:i], word[i+1:]
                    for j in 'abcdefghijklmnopqrstuvwxyz':
                        if word[i] != j:
                            nextword = part1 + j + part2
                            if nextword in wordlist:
                                current.add(nextword)
                                previousMap[nextword].append(word)
            if len(current) == 0: return res
            if endWord in current: break

        self.dfsBuildPath([], endWord, previousMap, res)
        return res

    def dfsBuildPath(self, path, endWord, previousMap, res):
        path.append(endWord)
        if len(previousMap[endWord]) == 0:
            currPath = path[:]
            currPath.reverse()
            res.append(currPath)
            return
        for w in previousMap[endWord]:
            self.dfsBuildPath(path, w, previousMap, res)
            path.pop()

    def minSubArrayLen(self, s, nums):
        n = len(nums)
        window = n+1
        start, end = 0, 0
        sum = 0
        while end < n:
            while end<n and sum<s:
                sum += nums[end]
                end += 1
            while start<end and sum>=s:
                window = min(window, end-start)
                sum -= nums[start]
                start += 1
        return [0, window][window<=n]

    def canJump(self, nums):
        def dfs(nums, start):
            steps = nums[start]
            if steps+start >= len(nums)-1: return True
            for x in xrange(start+1, start+steps+1):
                if dfs(nums,x): return True
            return False
        return dfs(nums, 0)

    def reverseBetween(self, head, m, n):
        if head==None or head.next==None: return head
        dummy = ListNode(0)
        dummy.next = head
        p = dummy
        for x in xrange(1,m):
            p = p.next
        q = p.next # q is the m-th node
        r = q.next
        for x in xrange(m, n):
            tmp = r.next
            r.next = q
            q = r
            r = tmp
        p.next.next = r
        p.next = q
        return dummy.next

    def isMatch(self, s, p): # RT: O(n), Space: O(1)
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        # the position of the last occurrence of * in p
        lastStartPos = -1
        # the position of the last character in s which matches with *
        lastCharMatchStar = 0
        idxPattern = idxStr = 0
        while idxStr < len(s):
            # when there is a match, continue to compare the characters in s and p
            if idxPattern<len(p) and (s[idxStr]==p[idxPattern] or p[idxPattern]=='?'):
                idxStr += 1
                idxPattern += 1
                continue
            # when * comes, save the positions of idxPattern and idxStr
            elif idxPattern<len(p) and p[idxPattern]=='*':
                lastStartPos = idxPattern
                lastCharMatchStar = idxStr
                idxPattern+=1
                continue
            # when there is a conflict, and we have a star, reset idxPattern
            # to point to the next position of previous, and also move ss
            # to its next position, and set idxStr to point to ss's position;
            # after these settings, continue to check
            elif lastStartPos!=-1:
                idxPattern = lastStartPos+1
                lastCharMatchStar += 1
                idxStr = lastCharMatchStar
                continue
            # if all the conditions are not satisfied, s does not match p.
            else: return False
        while idxPattern<len(p) and p[idxPattern]=='*':
            idxPattern += 1
        if idxPattern == len(p): return True
        return False

    def longestValidParentheses(self, s):
        maxlen = 0
        # store the indices of all unmatched left brackets
        stack = []
        # index of the last unmatched right bracket
        lastUnmatchedRight = -1
        for i in range(len(s)):
            if s[i]=='(': stack.append(i) # push the INDEX into the stack!!!!
            else:
                if stack == []:
                    lastUnmatchedRight = i
                else:
                    stack.pop()
                    if stack == []:
                        # current right bracket - the last unmatched right bracket
                        maxlen = max(maxlen, i - lastUnmatchedRight)
                    else:
                        # current right bracket - the last unmatched left bracket
                        maxlen = max(maxlen, i-stack[-1])
        return maxlen

    def recoverTree(self, root):
        """
        :type root: TreeNode
        :rtype: void Do not return anything, modify root in-place instead.
        """
        def inorderTraversal(root):
            res = []
            if root==None: return res
            if root.left: res += inorderTraversal(root.left)
            res.append(root)
            if root.right: res += inorderTraversal(root.right)
            return res

        if root:
            inorderNodes = inorderTraversal(root)
            x, n = 0, len(inorderNodes)
            while x<n-1:
                if x==0 and n==2:
                    if inorderNodes[x].val>inorderNodes[x+1].val:
                        inorderNodes[x].val,inorderNodes[x+1].val = inorderNodes[x+1].val,inorderNodes[x].val
                        break
                elif x>0:
                    if inorderNodes[x-1].val>inorderNodes[x].val and inorderNodes[x].val<inorderNodes[x+1].val:
                        inorderNodes[x-1].val,inorderNodes[x].val = inorderNodes[x].val,inorderNodes[x-1].val
                        break
                    elif inorderNodes[x-1].val<inorderNodes[x].val and inorderNodes[x].val>inorderNodes[x+1].val:
                        inorderNodes[x].val,inorderNodes[x+1].val = inorderNodes[x+1].val,inorderNodes[x].val
                        break
                    elif inorderNodes[x-1].val>inorderNodes[x].val>inorderNodes[x+1].val:
                        inorderNodes[x-1].val,inorderNodes[x+1].val = inorderNodes[x+1].val,inorderNodes[x-1].val
                        break
                x += 1
        return root

    def isSameTree(self, p, q):
        if p==None: return q==None
        if q==None: return p==None
        stack = [(p,q)]
        while stack!=[]:
            root1, root2 = stack.pop()
            if root1==None and root2==None: continue
            elif root1 and root2 and root1.val==root2.val:
                stack.append((root1.left, root2.left))
                stack.append((root1.right, root2.right))
            else: return False
        return True

    def flatten(self, root):
        p = root
        stack = []
        while p:
            if p.left==None and p.right==None:
                if stack!=[]:
                    p.right = stack.pop()
                    p = p.right
                else: break
            elif p.left and p.right:
                stack.append(p.right)
                p.right = p.left
                p.left = None
                p = p.right
            elif p.left:
                p.right = p.left
                p.left = None
                p = p.right
            elif p.right:
                p = p.right

    def restoreIpAddresses(self, s):
        """
        :type s: str
        :rtype: List[str]
        """
        def dfs(s, sub, ip):
            if sub==4:
                if s=='': ips.append(ip[1:])
                return
            for x in xrange(1, 4):
                if len(s)-x < 4-sub-1 or len(s)-x > (4-sub-1)*3: continue
                if len(s[:x])>1 and s[0]=='0': break
                if x==3 and int(s[:x])>255: break
                dfs(s[x:], sub+1, ip+'.'+s[:x])

        ips = []
        dfs(s, 0, '')
        return ips

    def maxSubArray(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        def getMaxArray(nums, left, right):
            if left>right: return -10000
            mid = (left+right)/2
            getMaxArray(nums, left, mid-1)
            getMaxArray(nums, mid+1, right)

            sum = 0; mlmax = 0
            for x in xrange(mid-1, left-1, -1):
                sum += nums[x]
                mlmax = max(mlmax, sum)
            sum = 0; mrmax = 0
            for x in xrange(mid, right+1):
                sum += nums[x]
                mrmax = max(mrmax, sum)
            self.maxsum = max(self.maxsum, mlmax+mrmax+nums[mid])

        n = len(nums)
        if n==0: return 0
        if len(nums)==1: return nums[0]
        self.maxsum = -10000
        return getMaxArray(nums, 0, n-1)

    def maximalRectangle(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if len(matrix)==0: return 0
        m, n = len(matrix), len(matrix[0])
        dp = [0 for _ in xrange(n)]
        maxarea = 0
        for i in xrange(m):
            for j in xrange(n):
                dp[j] = dp[j]+1 if matrix[i][j]=='1' else 0
            maxarea = max(maxarea, self.largestRectangleArea(dp))
        return maxarea

    def largestRectangleArea(self, height):
        stack=[]; area = 0
        i = 0
        while i<len(height):
            if stack==[] or height[i]>height[stack[-1]]:
                stack.append(i)
                i += 1
            else:
                curr = stack.pop()
                width = i if stack==[] else i-stack[-1]-1
                area = max(area, width * height[curr])

        while stack!=[]:
            curr = stack.pop()
            width = i if stack==[] else len(height)-stack[-1]-1
            area = max(area, width * height[curr])

        return area

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n==0: return 0
        # the profits gained in the first i days
        profits1 = [0 for _ in range(n)]
        # the profits gained after i days
        profits2 = [0 for _ in range(n)]

        # compute the profits gained in the first transaction
        minprice = prices[0]
        profits1[0] = 0
        for i in range(1,n):
            profits1[i] = max(profits1[i-1], prices[i]-minprice)
            minprice = min(minprice, prices[i])

        # compute the profits gained in the second transaction
        maxprice = prices[n-1]
        profits2[n-1] = 0
        for i in range(n-2,-1,-1):
            profits2[i] = max(profits2[i+1], maxprice-prices[i])
            maxprice = max(maxprice, prices[i])

        # merge the profits gained in these two transactions
        maxprofit = 0
        for i in range(n):
            maxprofit = max(maxprofit, profits1[i]+profits2[i])

        return maxprofit

    def isScramble(self, s1, s2):
        """
        :type s1: str
        :type s2: str
        :rtype: bool
        """
        if len(s1)!=len(s2): return False
        n = len(s1)
        dp = [[[False for k in xrange(n+1)] for j in xrange(n)] for i in xrange(n)]
        for k in xrange(1, n+1):
            for i in xrange(n+1-k):
                for j in xrange(n+1-k):
                    if k==1: dp[i][j][k] = s1[i]==s2[j]
                    else:
                        for l in xrange(1,k):
                            if dp[i][j][k]: break
                            else:
                                dp[i][j][k] = dp[i][j][l] and dp[i+l][j+l][k-l] or dp[i][j+k-l][l] and dp[i+l][j][k-l]
        return dp[0][0][n]

    def numDecodings(self, s):
        """
        :type s: str
        :rtype: int
        """
        if s=="" or s[0]=='0': return 0
        dp = [1,1]
        for i in range(2, len(s)+1):
            if 10 <= int(s[i-2:i]) <=26 and s[i-1]!='0':
                dp.append(dp[i-2]+dp[i-1])
            elif int(s[i-2:i])==10 or int(s[i-2:i])==20:
                dp.append(dp[i-2])
            elif s[i-1]!='0':
                dp.append(dp[i-1])
            else:
                return 0

        return dp[len(s)]

    def wordBreak(self, s, wordDict):
        n = len(s)
        dp = [[False for j in xrange(n)] for i in xrange(n)]
        for l in xrange(1, n+1):
            for i in xrange(n-l):
                j = i+l-1
                if s[i:j+1] in wordDict:
                    dp[i][j] = True
                else:
                    for k in xrange(i+1,j+1):
                        if dp[i][k-1] and dp[k][j]:
                            dp[i][j] = True
                            break
        return dp[0][n-1]


    def isMatch(self, s, p): # RT: O(n), Space: O(1)
        """
        :type s: str
        :type p: str
        :rtype: bool
        """
        if len(p) - p.count('*') > len(s): return False

        # replace multiple * with one *
        # e.g., a***b**c => a*b*c
        pattern = []
        isFirst = True
        for i in range(len(p)):
            if p[i]=='*':
                if isFirst:
                    pattern.append(p[i])
                    isFirst = False
            else:
                pattern.append(p[i])
                isFirst = True

        # the position of the last occurrence of * in p
        lastStartPos = -1
        # the position of the last character in s which matches with *
        lastCharMatchStar = 0
        idxp = idxs = 0
        while idxs < len(s):
            # when there is a match, continue to compare the characters in s and p
            if idxp<len(pattern) and (s[idxs]==pattern[idxp] or pattern[idxp]=='?'):
                idxs += 1
                idxp += 1
                continue
            # when * comes, save the positions of idxp and idxs
            elif idxp<len(pattern) and pattern[idxp]=='*':
                lastStartPos = idxp
                lastCharMatchStar = idxs
                idxp+=1
                continue
            # when there is a conflict, and we have a star, reset idxp
            # to point to the next position of previous, and also move ss
            # to its next position, and set idxs to point to ss's position;
            # after these settings, continue to check
            elif lastStartPos!=-1:
                idxp = lastStartPos+1
                lastCharMatchStar += 1
                idxs = lastCharMatchStar
                continue
            # if all the conditions are not satisfied, s does not match p.
            else: return False

        if (idxp == len(pattern)) or (idxp<len(pattern) and pattern[idxp]=='*'): return True
        return False

    def nthUglyNumber(self, n):
        """
        :type n: int
        :rtype: int
        """
        if n==0: return 0
        if n==1: return 1
        target = 1
        x2, x3, x5 = [], [], []
        for i in range(n-1):
            x2.append(target*2)
            x3.append(target*3)
            x5.append(target*5)
            target = min(x2[0],x3[0],x5[0])

            # remove the duplicates
            if target==x2[0]: x2.pop(0)
            if target==x3[0]: x3.pop(0)
            if target==x5[0]: x5.pop(0)
        return target

    def numSquares(self, n): # TLE
        while n%4==0: n/=4
        if n%8==7: return 4
        m = int(n**0.5)+1
        for x in xrange(m):
            y = int((n-x*x)**0.5)
            if x*x + y*y == n:
                if x>0 and y>0:
                    return 2
                else: return 1
        return 3

    def maximalSquare(self, matrix):
        """
        :type matrix: List[List[str]]
        :rtype: int
        """
        if matrix==[]: return 0
        m = len(matrix); n = len(matrix[0])
        dp = [[0] * n for i in range(m)]
        width = 0
        for i in range(m):
            for j in range(n):
                dp[i][j] = int(matrix[i][j])
                if i and j and dp[i][j]:
                    dp[i][j] = min(dp[i-1][j], dp[i][j-1], dp[i-1][j-1]) + 1
                width = max(width, dp[i][j])
        return width*width

    def lengthOfLIS_dp(self, nums): # RT: O(nlogn)
        """
        :type nums: List[int]
        :rtype: int
        """
        n = len(nums)
        if n==0: return n
        dp = [0 for _ in range(n)]
        length = 0
        for i in range(1, n):
            if nums[dp[0]] > nums[i]:
                dp[0] = i
            elif nums[dp[length]] < nums[i]:
                length += 1
                dp[length] = i
            else:
                # do a binary search to find the ceiling
                # of nums[i] and put it there
                index = self.getCeilIndex(nums, dp, length, nums[i])
                dp[index] = i
        return length+1

    def getCeilIndex(self, nums, dp, length, val): # binary search: O(logn)
        start = 0; end = length
        while start <= end:
            mid = (start+end)/2
            if mid<length and nums[dp[mid]]<val<=nums[dp[mid+1]]:
                return mid+1
            elif val < nums[dp[mid]]:
                end = mid - 1
            else:
                start = mid + 1
        return -1

    def robcircle(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if nums == None or len(nums) == 0: return 0
        elif len(nums) == 1: return nums[0]
        # As the houses form a circle, if rob the 1st house,
        # you cannot rob the last one, so need 2 dp scan.
        return max(self.roblinear(nums[:len(nums)-1]), self.roblinear(nums[1:]))

    def roblinear(self, nums):
        n = len(nums)
        if n==1: return nums[0]
        dp = [nums[0]]
        dp.append(max(nums[1], dp[0]))
        for i in range(2, n):
            dp.append(max(dp[i-1], dp[i-2]+nums[i]))

        return dp[n-1]

    def coinChange(self, coins, amount):
        coins.sort()
        if amount==0: return 0
        if amount < coins[0]: return -1
        dp = [-1 for _ in xrange(amount+1)]
        candidates = []
        for c in coins:
            if c <= amount:
                dp[c] = 1
                candidates.append(c)
        for i in xrange(candidates[0]+1, amount+1):
            if i not in coins:
                minvalue = -1
                for coin in candidates:
                    if i>=coin and dp[i-coin]!=-1:
                        if minvalue==-1: minvalue=dp[i-coin]
                        else: minvalue = min(minvalue, dp[i-coin])
                dp[i] = 1 + minvalue if minvalue > 0 else minvalue

        return dp[-1]

    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        newnums = [1] + nums + [1]
        n = len(newnums)
        dp = [[0]*n for _ in range(n)]
        for k in range(2,n): # k is the length of span
            for l in range(n-k): # l is the left bound
                r = l+k # r is the right bount
                for m in range(l+1,r):
                    dp[l][r] = max(dp[l][r], dp[l][m]+newnums[l]*newnums[m]*newnums[r]+dp[m][r])
        return dp[0][n-1]

    def maxProfitDP(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        size = len(prices)
        if size < 2:
            return 0
        holdDP = [None] * size
        notHoldDP = [None] * size
        notHoldDP[0], notHoldDP[1] = 0, max(0, prices[1] - prices[0])
        holdDP[0], holdDP[1] = -prices[0], max(-prices[0], -prices[1])
        for x in range(2, size):
            notHoldDP[x] = max(notHoldDP[x-1], holdDP[x - 1] + prices[x])
            holdDP[x] = max(holdDP[x-1], notHoldDP[x - 2] - prices[x])
        return notHoldDP[-1]

    def longestConsecutive_new(self, nums):
        if len(nums) <= 1: return len(nums)
        dict = {}
        for i in xrange(len(nums)):
            dict[nums[i]] = i
        keys = dict.keys()
        i = 0
        lcs = 0
        while i < len(keys):
            key = keys[i]
            l = 0
            while dict.has_key(key):
                l += 1
                key += 1
                i += 1
            lcs = max(lcs, l)
        return lcs

    def threeSumdfs(self, nums):
        """
        :type nums: List[int]
        :rtype: List[List[int]]
        """

        def dfs(depth, target, nums, valuelist):
            if depth == 0:
                if target == 0: res.add(tuple(valuelist))
                return
            if target<nums[0] or target>nums[len(nums)-1]: return
            for i in xrange(len(nums)-depth+1):
                dfs(depth - 1, target - nums[i], nums[i + 1:], valuelist + [nums[i]])

        nums.sort()
        res = set()
        dfs(3, 0, nums, [])
        return [list(t) for t in res]

    def singleNumber(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        # the result is a 32-bit integer
        res = 0
        negatives = 0
        for x in xrange(32):
            count = 0
            for i in xrange(len(nums)):
                if nums[i] < 0:
                    nums[i] = ~(nums[i]+1)
                    negatives += 1
                if (nums[i] >> x) & 1 == 1:
                    count += 1
            bit = count % 3
            if bit == 1:
                res = res | (bit << x)
        return res if negatives % 3 == 0 else -res

    def insert(self, intervals, newInterval):
        n = len(intervals)
        ni = newInterval
        if n == 0: return [ni]
        if ni.end < intervals[0].start:
            return [ni] + intervals
        elif ni.start > intervals[-1].end:
            return intervals + [ni]

        res = []
        i = 0
        while i < len(intervals):
            if ni.start < intervals[i].start <= ni.end:
                j = i
                while j < n:
                    if intervals[j].start <= ni.end <= intervals[j].end:
                        ni.end = intervals[j].end
                        res.append(ni)
                        break
                    j += 1
                if j == n:
                    res.append(ni)
                    break
                i = j + 1
            elif ni.start <= intervals[i].end <= ni.end:
                j = i
                while j < n:
                    if intervals[j].start <= ni.end <= intervals[j].end:
                        ni.start = intervals[i].start
                        ni.end = max(ni.end, intervals[j].end)
                        res.append(ni)
                        break
                    j += 1
                if j == n:
                    ni.start = intervals[i].start
                    res.append(ni)
                    break
                i = j + 1
            else:
                res.append(intervals[i])
                i += 1
        return res

    def addTwoNumbers(self, l1, l2):  # RT: O(max{len(l1),len(l2)})
        """
        :type l1: ListNode
        :type l2: ListNode
        :rtype: ListNode
        """
        if l1 == None: return l2
        if l2 == None: return l1

        h1 = l1
        h2 = l2
        while h1 != None and h2 != None:
            h1 = h1.next
            h2 = h2.next

        if h2 != None: l1, l2 = l2, l1

        head = l1
        carry = 0
        while l2 != None:
            asum = l1.val + l2.val + carry
            l1.val = asum % 10
            carry = asum / 10
            # l1 and l2 have same size
            if l2.next is None and l1.next is None and carry == 1:
                l1.next = ListNode(carry)
                carry = 0
            l1 = l1.next
            l2 = l2.next

        while carry == 1 and l1:
            asum = carry + l1.val
            l1.val = asum % 10
            carry = asum / 10
            if l1.next is None and carry == 1:
                l1.next = ListNode(carry)
                break
            l1 = l1.next
        return head

    def restoreIpAddresses_new(self, s):
        def dfs(k, s, address):
            if not ((4 - k) <= len(s) <= (4 - k) * 3): return
            if k == 3:
                if ((1 < len(s) <= 3 and s[0] != '0') or len(s) == 1) and 0 <= int(s) <= 255:
                    res.append(address[1:] + '.' + s)
                return
            for l in xrange(1, 4):
                if l > 1 and s[0] == '0': return
                if 0 <= int(s[:l]) <= 255:
                    dfs(k + 1, s[l:], address + '.' + s[:l])

        res = []
        dfs(0, s, '')
        return res

    def solveSudoku(self, board):
        """
        :type board: List[List[str]]
        :rtype: void Do not return anything, modify board in-place instead.
        """
        pos, H, V, G = [], [0] * 9, [0] * 9, [0] * 9  # Empty cells'position,horizontal,vertical,grid
        ctoV = {str(i): 1 << (i - 1) for i in range(1, 10)}  # eg:'4'=>1000
        self.vtoC = {1 << (i - 1): str(i) for i in range(1, 10)}  # eg:100=>'3'
        for i, row in enumerate(board):
            for j, cell in enumerate(row):
                if cell != '.':
                    v = ctoV[cell]
                    H[i], V[j], G[i / 3 * 3 + j / 3] = H[i] | v, V[j] | v, G[i / 3 * 3 + j / 3] | v
                else:
                    pos += (i, j),
        # dict {(i,j):[possible vals(bit-identify),count]}
        posDict = {(i, j): [x, self.countOnes(x)] for i, j in pos \
                   for x in [0x1ff & ~(H[i] | V[j] | G[i / 3 * 3 + j / 3])]}
        self.solve(board, posDict)

    def countOnes(self, n):
        count = 0
        while n:
            count, n = count + 1, n & ~(n & (~n + 1))
        return count

    def solve(self, board, posDict):
        if len(posDict) == 0:
            return True
        # sort posDict according to the number of '1' in V, and get the minimum.
        p = min(posDict.keys(), key=lambda x: posDict[x][1])
        candidate = posDict[p][0]
        while candidate:
            v = candidate & (~candidate + 1)  # get last '1'
            candidate &= ~v # remove the last '1'
            tmp = self.update(board, posDict, p, v)  # update board and posDict
            if self.solve(board, posDict):  # solve next position
                return True
            self.rollback(board, posDict, p, v, tmp)  # restore the original state
        return False

    def update(self, board, posDict, p, v):
        i, j = p[0], p[1]
        board[i][j] = self.vtoC[v]
        tmp = [posDict[p]]
        del posDict[p]
        for key in posDict.keys():
            if i == key[0] or j == key[1] or (i / 3, j / 3) == (key[0] / 3, key[1] / 3):  # relevant points
                if posDict[key][0] & v:  # need modify
                    posDict[key][0] &= ~v
                    posDict[key][1] -= 1
                    tmp += key,  # Record these points.
        return tmp

    def rollback(self, board, posDict, p, v, tmp):
        board[p[0]][p[1]] = '.'
        posDict[p] = tmp[0]
        for key in tmp[1:]:
            posDict[key][0] |= v
            posDict[key][1] += 1

    def buildTable(self, s):
        n = len(s)
        table = [[False] * n for _ in xrange(n)]
        for length in xrange(1,n+1):
            for i in xrange(n-length+1):
                j = (i+length-1)%n
                if i==j: table[i][j] = True
                elif i==j-1: table[i][j] = s[i]==s[j]
                else: table[i][j] = (s[i]==s[j]) and table[i + 1][j - 1]
        return table

    def partition(self, s):
        """
        :type s: str
        :rtype: List[List[str]]
        """
        def dfs(s, offset, valuelist):
            if len(s) == 0:
                res.append(valuelist)
                return
            # try each possible sub-string
            for l in range(1, len(s) + 1):
                # if current substring is palindrome,
                # recursively check if the remaining
                # substring contains more palindrome strings
                if table[0+offset][l-1+offset]:
                    dfs(s[l:], offset+l, valuelist + [s[:l]])

        def buildTable(s):
            n = len(s)
            table = [[False] * n for _ in xrange(n)]
            for length in xrange(1, n + 1):
                for i in xrange(n - length + 1):
                    j = (i + length - 1) % n
                    if i == j:
                        table[i][j] = True
                    elif i == j - 1:
                        table[i][j] = s[i] == s[j]
                    else:
                        table[i][j] = (s[i] == s[j]) and table[i + 1][j - 1]
            return table

        res = []
        table = buildTable(s)
        dfs(s, 0, [])
        return res

    def minCut(self, s):
        """
        :type s: str
        :rtype: int
        """
        n = len(s)

        # build DP table
        T = [[False] * n for _ in xrange(n)]
        for l in xrange(1, n + 1):
            for i in xrange(n - l + 1):
                j = i + l - 1
                if i == j:
                    T[i][j] = True
                elif i == j - 1:
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
            cuts[i] = min(cuts[i], cuts[i-1]+1)
        return cuts[-1]

    def lengthOfLIS(self, nums):
        n = len(nums)
        if n == 0: return n
        dp = [0 for _ in range(n)]
        dp[0] = 1
        for i in range(1, n):
            if nums[i] <= nums[i - 1]:
                dp[i] = dp[i - 1]
            else:
                dp[i] = 1 + dp[i - 1]
        return dp[n - 1]

    def maxProfit(self, prices):
        """
        :type prices: List[int]
        :rtype: int
        """
        n = len(prices)
        if n == 0: return 0
        # the profits gained in the first i days
        profits1 = [0 for _ in range(n)]
        # the profits gained after i days
        profits2 = [0 for _ in range(n)]

        # compute the profits gained in the first transaction
        minprice = prices[0]
        profits1[0] = 0
        for i in range(1, n):
            profits1[i] = max(profits1[i - 1], prices[i] - minprice)
            minprice = min(minprice, prices[i])

        # compute the profits gained in the second transaction
        maxprice = prices[n - 1]
        profits2[n - 1] = 0
        for i in range(n - 2, -1, -1):
            profits2[i] = max(profits2[i + 1], maxprice - prices[i])
            maxprice = max(maxprice, prices[i])

        # merge the profits gained in these two transactions
        maxprofit = 0
        for i in range(n):
            maxprofit = max(maxprofit, profits1[i] + profits2[i])
        return maxprofit

    def maxProfit4(self, k, prices):
        """
        :type k: int
        :type prices: List[int]
        :rtype: int
        """
        size = len(prices)
        if k > size / 2:
            return self.quickSolve(size, prices)
        dp = [None] * (2 * k + 1)
        dp[0] = 0
        for i in range(size):
            for j in range(1, min(2 * k, i + 1) + 1):
                dp[j] = max(dp[j], dp[j - 1] + prices[i] * [1, -1][j % 2])
        return dp[2 * k]

    def quickSolve(self, size, prices):
        sum = 0
        for x in range(size - 1):
            if prices[x + 1] > prices[x]:
                sum += prices[x + 1] - prices[x]
        return sum

    def maxCoins(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        newnums = [1] + nums + [1]
        n = len(newnums)
        dp = [[0] * n for _ in range(n)]
        # the length of each span
        for length in range(2, n):
            # the left bound of the range to check
            for start in range(n - length):
                # the right bound of the range to check
                end = start + length
                for lastBalloon in range(start + 1, end):
                    dp[start][end] = max(dp[start][end],
                                         dp[start][lastBalloon] + newnums[start] * newnums[lastBalloon] * newnums[end] +
                                         dp[lastBalloon][end])
        return dp[0][n - 1]





















