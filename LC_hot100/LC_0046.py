"""
给定一个不含重复数字的数组 nums ，返回其 所有可能的全排列 。你可以 按任意顺序 返回答案。

输入：nums = [1,2,3]
输出：[[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]]

输入：nums = [0,1]
输出：[[0,1],[1,0]]

输入：nums = [1]
输出：[[1]]
"""


from typing import List


class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        path = []
        used = [False] * len(nums)
        def dfs(path, used, res):
            if len(path) == len(nums):
                res.append(path.copy())
                print(path)
                return
            for p in range(len(nums)):
                if used[p]:
                    continue
                path.append(nums[p])
                used[p] = True
                dfs(path, used, res)
                used[p] = False
                path.pop(-1)
        dfs(path, used, res)
        return res

if __name__ == '__main__':
    nums = [1, 2, 3]
    solution = Solution()
    res = solution.permute(nums)
    print(res)
