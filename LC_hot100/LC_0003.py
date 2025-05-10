"""
给定一个字符串 s ，请你找出其中不含有重复字符的 最长 子串 的长度。

 

示例 1:

输入: s = "abcabcbb"
输出: 3 
解释: 因为无重复字符的最长子串是 "abc"，所以其长度为 3。
示例 2:

输入: s = "bbbbb"
输出: 1
解释: 因为无重复字符的最长子串是 "b"，所以其长度为 1。
示例 3:

输入: s = "pwwkew"
输出: 3
解释: 因为无重复字符的最长子串是 "wke"，所以其长度为 3。
     请注意，你的答案必须是 子串 的长度，"pwke" 是一个子序列，不是子串。
 

提示：

0 <= s.length <= 5 * 104
s 由英文字母、数字、符号和空格组成
"""

class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        wind = set()
        p, q, maxn = 0, 0, 0
        while p < len(s):
            while q < len(s) and s[q] not in wind:
                wind.add(s[q])
                q += 1
            if len(wind) > maxn:
                maxn = len(wind)
            wind.remove(s[p])
            p += 1    
        return maxn

if __name__ == "__main__":
    Solution().lengthOfLongestSubstring("abcabcbb")