
"""
    给定一个整数数组 nums 和一个整数目标值 target，请你在该数组中找出 和为目标值 target  的那 两个 整数，并返回它们的数组下标。
    你可以假设每种输入只会对应一个答案，并且你不能使用两次相同的元素。
    你可以按任意顺序返回答案。
"""

from typing import List


class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        hist = dict()
        result = []
        for i in range(len(nums)):
            if target - nums[i] in hist:
                result.append(i)
                result.append(hist[target - nums[i]])
                break
            else:
                hist[nums[i]] = i
        return result
        
