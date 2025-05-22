"""
给你一个整数数组 nums ，判断是否存在三元组 [nums[i], nums[j], nums[k]] 满足 i != j、i != k 且 j != k ，
同时还满足 nums[i] + nums[j] + nums[k] == 0 。请你返回所有和为 0 且不重复的三元组。
注意：答案中不可以包含重复的三元组。
"""


from typing import List


class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        n = len(nums)
        nums = sorted(nums)
        print(nums)
        res = []
        for i in range(n - 2):
            if i > 0 and nums[i] == nums[i - 1]:
                continue
            p, q = i+1, n-1
            while p < q:
                if nums[p] + nums[q] + nums[i] < 0:
                    p += 1
                elif nums[p] + nums[q] + nums[i] > 0:
                    q -= 1
                else:
                    res.append([nums[i], nums[p], nums[q]])
                    while p < q and nums[p + 1] == nums[p]:
                        p += 1
                    p += 1
                    while p < q and nums[q - 1] == nums[q]:
                        q -= 1
                    q -= 1
        return res
if __name__ == "__main__":
    print(Solution().threeSum([-1,0,1,2,-1,-4]))