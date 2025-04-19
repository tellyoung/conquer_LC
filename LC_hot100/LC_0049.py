"""
    给你一个字符串数组，请你将 字母异位词 组合在一起。可以按任意顺序返回结果列表。
    字母异位词 是由重新排列源单词的所有字母得到的一个新单词。

    输入: strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    输出: [["bat"],["nat","tan"],["ate","eat","tea"]]
"""

from typing import List


class Solution:
    def groupAnagrams(self, strs: List[str]) -> List[List[str]]:
        result = []
        group = dict()

        for str in strs:
            k = "".join(sorted(str))
            if k in group:
                group[k].append(str)
            else:
                group[k] = [str]
        
        for k in group.keys():
            result.append(group[k])
        
        return result

if __name__ == "__main__":
    strs = ["eat", "tea", "tan", "ate", "nat", "bat"]
    Solution().groupAnagrams(strs)