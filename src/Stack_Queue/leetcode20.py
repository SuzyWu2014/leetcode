

class Solution(object):
    def isValid(self, s):
        """
        :type s: str
        :rtype: bool
        """
        stack = []
        for x in s:
            if x in '([{': stack.append(x)
            else:
                if stack==[]: return False
                top = stack.pop()
                if (top == '(' and x == ')') or \
                   (top == '[' and x == ']') or \
                   (top == '{' and x == '}'):
                    continue
                else:
                    return False
        if stack != []:
            return False
        return True
