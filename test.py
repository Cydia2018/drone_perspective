import itertools


class Solution:
    def lengthOfLIS(self, nums):
        """
        :type nums: List[int]
        :rtype: int
        """
        if not nums:
            return 0

        len_nums = len(nums)
        mem = [1] * len_nums
        result = 1
        for i in range(1, len_nums):
            for j in range(i):
                if nums[j] < nums[i]:
                    mem[i] = max(mem[i], 1 + mem[j])

            result = max(result, mem[i])

        return result


if __name__ == '__main__':
    a=[0,1,2,5,9,7,8]
    print(Solution().lengthOfLIS(a))