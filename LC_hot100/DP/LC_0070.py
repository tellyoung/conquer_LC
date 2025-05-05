"""
假设你正在爬楼梯。需要 n 阶你才能到达楼顶。
每次你可以爬 1 或 2 个台阶。你有多少种不同的方法可以爬到楼顶呢？

输入：n = 2
输出：2
解释：有两种方法可以爬到楼顶。
1. 1 阶 + 1 阶
2. 2 阶

输入：n = 3
输出：3
解释：有三种方法可以爬到楼顶。
1. 1 阶 + 1 阶 + 1 阶
2. 1 阶 + 2 阶
3. 2 阶 + 1 阶
"""

class Solution:
    def climbStairs2(self, n: int) -> int:
        path = []
        res = [0]
        s_tmp = 0

        def dfs(path, s_tmp, res):
            if s_tmp == n:
                res[0] += 1
                # print(res[0])
                print(path)
                return
            if s_tmp > n:
                return
            for i in range(1,3):
                path.append(i)
                s_tmp += i
                dfs(path, s_tmp, res)
                s_tmp -= i
                path.pop(-1)
        dfs(path, s_tmp, res)
        print(res[0])
        return res[0]

    # 记忆化递归，自顶向下
    def climbStairs(self, n: int) -> int:
        def dfs(i, memo):
            if i == 0 or i == 1:
                return 1
            if memo[i] == -1:
                memo[i] = dfs(i - 1, memo) + dfs(i - 2, memo)
            return memo[i]

        # memo: [-1] * (n - 1)
        # -1 表示没有计算过，最大索引为 n，因此数组大小需要 n + 1
        return dfs(n, [-1] * (n + 1))

if __name__ == "__main__":
    Solution().climbStairs(7)
