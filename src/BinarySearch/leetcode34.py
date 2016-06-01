
class Solution(object):
    def searchRange_BS(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: List[int]
        """
        begin = end = -1
        # find the first occurrence of target in nums
        front, back = 0, len(nums)-1
        while front <= back:
            mid = (front+back)/2
            if nums[mid ] < target:
                front = mid+1
            elif nums[mid] > target:
                back = mid - 1
            else:
                begin = mid
                back = mid - 1

        # if begin is -1, it means there is no target in nums
        if begin == -1:
            return [-1, -1]

        # find the last occurrence of target in nums
        front, back = begin, len(nums) - 1
        while front <= back:
            mid = (front + back)/2
            if nums[mid] < target:
                front = mid + 1
            elif nums[mid] > target:
                back = mid - 1
            else:
                end = mid
                front = mid + 1
        return [begin, end]

    def searchRange_recursive(self, nums, target):
        def search(start, end):
            if nums[start] == target == nums[end]:
                return [start, end]
            if nums[start] <= target <= nums[end]:
                mid = (start + end) / 2
                left, right = search(start, mid), search(mid + 1, end)

                if -1 in left + right:
                    # target range in left side or right side
                    return max(left, right)
                else:
                    # target range across left side and right side
                    return [left[0], right[1]]
            return [-1, -1]

        return search(0, len(nums) - 1)
