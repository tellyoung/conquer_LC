"""
    给定一个未排序的整数数组 nums ，找出数字连续的最长序列（不要求序列元素在原数组中连续）的长度。
    请你设计并实现时间复杂度为 O(n) 的算法解决此问题。


    输入：nums = [100,4,200,1,3,2]
    输出：4
    解释：最长数字连续序列是 [1, 2, 3, 4]。它的长度为 4。
"""

from typing import List


class Solution:
    def longestConsecutive(self, nums: List[int]) -> int:
        nums_set = set(nums)
        nums_sorted = sorted(list(nums_set))

        cnt = 0
        max_cnt = 0
        for i in range(len(nums_sorted)):
            if cnt == 0 or nums_sorted[i] - 1 in nums_set:# or nums_sorted[i] == nums_sorted[i-1]:
                cnt += 1
                max_cnt = cnt if cnt > max_cnt else max_cnt
            else:
                cnt = 1
        return max_cnt




if __name__ == "__main__":
    Solution().longestConsecutive([0,3,7,2,5,8,4,6,0,1])