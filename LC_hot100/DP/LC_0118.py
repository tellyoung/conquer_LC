"""
给定一个非负整数 numRows，生成「杨辉三角」的前 numRows 行。
在「杨辉三角」中，每个数是它左上方和右上方的数的和。

[1]
[1,1]
[1,2,1]
[1,3,3,1]
[1,4,6,4,1]
"""

from typing import List


class Solution:
    def generate(self, numRows: int) -> List[List[int]]:
        res = [[1] * (i + 1) for i in range(numRows)]
        
        for i in range(2, numRows):
            for j in range(1, i):
                res[i][j] = res[i - 1][j - 1] + res[i - 1][j]
        
        print(res)

        return res
if __name__ == "__main__":
    Solution().generate(7)