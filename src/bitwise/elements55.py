
class Solution(object):
    """
    use bitwise operations >>, <<, &, |, ~, ^ to mimic multiplication
    and addition of two positive integers.
    """
    def multiply_bitwise(self, x, y):
        sum = 0
        while x:
            if x & 1:
                sum = self.add_bitwise(sum, y)
            x >> 1
            y << 1
        return sum

    def add_bitwise(self, a, b):
        sum = 0
        tmp_a = a
        tmp_b = b
        carryin = 0
        k = 1
        while tmp_a or tmp_b:
            ak = a & k
            bk = b & k
            carryout = (ak & bk) | (ak & carryin) | (bk & carryin)
            sum |= ak ^ bk ^ carryin
            carryin = carryout << 1
            tmp_a >> 1
            tmp_b >> 1
            k << 1
        return sum | carryin
