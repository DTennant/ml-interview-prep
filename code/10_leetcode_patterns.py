"""LeetCode-style patterns: one canonical problem per pattern, with tests.

Pattern -> problem (LC number)
 1. Hashing            two_sum (1), group_anagrams (49), subarray_sum_k (560, prefix sum + hashmap)
 2. Sliding window     longest_unique_substring (3), min_window (76)
 3. Two pointers       three_sum (15), max_area (11)
 4. Binary search      search_rotated (33), min_eating_speed (875, binary search on answer)
 5. Heap               top_k_frequent (347), merge_k_sorted (23), MedianFinder (295)
 6. Monotonic stack    daily_temperatures (739), largest_rectangle (84)
 7. Intervals          merge_intervals (56), min_meeting_rooms (253)
 8. Graph BFS/DFS      num_islands (200), shortest_path_grid (1091)
 9. Topological sort   can_finish / course_order (207/210)
10. Union-Find         count_components (323)
11. Dijkstra           network_delay (743)
12. Trie               Trie (208)
13. Backtracking       subsets (78), permutations (46), combination_sum (39)
14. DP                 coin_change (322), lis (300, O(n log n)), edit_distance (72), lcs (1143)
15. Design             LRUCache (146, dict + doubly linked list)
16. ML-flavoured       reservoir_sample, weighted_sample (528), sparse_matmul, kth_largest (215, quickselect)

Interview checklist:
- state brute force + complexity first, then optimise; say time AND space complexity at the end
- walk through a tiny example by hand before coding; test edge cases: empty, single element, duplicates, negatives
"""
import bisect
import heapq
import random
from collections import Counter, OrderedDict, defaultdict, deque


# ---------------- 1. Hashing ----------------
def two_sum(nums, target):                       # O(n)
    seen = {}
    for i, x in enumerate(nums):
        if target - x in seen:
            return [seen[target - x], i]
        seen[x] = i
    return []


def group_anagrams(words):                       # O(n * k)
    g = defaultdict(list)
    for w in words:
        key = [0] * 26
        for c in w:
            key[ord(c) - 97] += 1
        g[tuple(key)].append(w)
    return list(g.values())


def subarray_sum_k(nums, k):                     # count subarrays summing to k, O(n)
    count, s, seen = 0, 0, Counter({0: 1})
    for x in nums:
        s += x
        count += seen[s - k]
        seen[s] += 1
    return count


# ---------------- 2. Sliding window ----------------
def longest_unique_substring(s):                 # O(n)
    last, left, best = {}, 0, 0
    for r, c in enumerate(s):
        if c in last and last[c] >= left:
            left = last[c] + 1
        last[c] = r
        best = max(best, r - left + 1)
    return best


def min_window(s, t):                            # O(|s| + |t|)
    need, missing = Counter(t), len(t)
    left, best = 0, (0, float("inf"))
    for r, c in enumerate(s):
        if need[c] > 0:
            missing -= 1
        need[c] -= 1
        while missing == 0:
            if r - left < best[1] - best[0]:
                best = (left, r)
            need[s[left]] += 1
            if need[s[left]] > 0:
                missing += 1
            left += 1
    return "" if best[1] == float("inf") else s[best[0]:best[1] + 1]


# ---------------- 3. Two pointers ----------------
def three_sum(nums):                             # O(n^2)
    nums, res = sorted(nums), []
    for i in range(len(nums) - 2):
        if i and nums[i] == nums[i - 1]:
            continue
        l, r = i + 1, len(nums) - 1
        while l < r:
            s = nums[i] + nums[l] + nums[r]
            if s < 0:
                l += 1
            elif s > 0:
                r -= 1
            else:
                res.append([nums[i], nums[l], nums[r]])
                while l < r and nums[l] == nums[l + 1]:
                    l += 1
                l, r = l + 1, r - 1
    return res


def max_area(h):                                 # O(n): always move the shorter side
    l, r, best = 0, len(h) - 1, 0
    while l < r:
        best = max(best, (r - l) * min(h[l], h[r]))
        if h[l] < h[r]:
            l += 1
        else:
            r -= 1
    return best


# ---------------- 4. Binary search ----------------
def search_rotated(nums, target):                # O(log n)
    lo, hi = 0, len(nums) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        if nums[mid] == target:
            return mid
        if nums[lo] <= nums[mid]:                # left half sorted
            if nums[lo] <= target < nums[mid]:
                hi = mid - 1
            else:
                lo = mid + 1
        else:                                    # right half sorted
            if nums[mid] < target <= nums[hi]:
                lo = mid + 1
            else:
                hi = mid - 1
    return -1


def min_eating_speed(piles, h):                  # binary search on the answer: O(n log max)
    lo, hi = 1, max(piles)
    while lo < hi:
        mid = (lo + hi) // 2
        if sum((p + mid - 1) // mid for p in piles) <= h:
            hi = mid
        else:
            lo = mid + 1
    return lo


# ---------------- 5. Heap ----------------
def top_k_frequent(nums, k):                     # O(n log k)
    return [x for _, x in heapq.nlargest(k, ((c, x) for x, c in Counter(nums).items()))]


def merge_k_sorted(lists):                       # O(N log k)
    h = [(lst[0], i, 0) for i, lst in enumerate(lists) if lst]
    heapq.heapify(h)
    out = []
    while h:
        v, i, j = heapq.heappop(h)
        out.append(v)
        if j + 1 < len(lists[i]):
            heapq.heappush(h, (lists[i][j + 1], i, j + 1))
    return out


class MedianFinder:                              # two heaps, O(log n) add, O(1) median
    def __init__(self):
        self.lo, self.hi = [], []                # lo: max-heap (negated), hi: min-heap

    def add(self, x):
        heapq.heappush(self.lo, -x)
        heapq.heappush(self.hi, -heapq.heappop(self.lo))
        if len(self.hi) > len(self.lo):
            heapq.heappush(self.lo, -heapq.heappop(self.hi))

    def median(self):
        return -self.lo[0] if len(self.lo) > len(self.hi) else (-self.lo[0] + self.hi[0]) / 2


# ---------------- 6. Monotonic stack ----------------
def daily_temperatures(t):                       # O(n)
    res, st = [0] * len(t), []
    for i, x in enumerate(t):
        while st and t[st[-1]] < x:
            j = st.pop()
            res[j] = i - j
        st.append(i)
    return res


def largest_rectangle(heights):                  # O(n)
    st, best = [], 0
    for i, h in enumerate(heights + [0]):
        start = i
        while st and st[-1][1] >= h:
            j, hh = st.pop()
            best = max(best, hh * (i - j))
            start = j
        st.append((start, h))
    return best


# ---------------- 7. Intervals ----------------
def merge_intervals(iv):                         # O(n log n)
    out = []
    for s, e in sorted(iv):
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def min_meeting_rooms(iv):                       # O(n log n)
    ends = []
    for s, e in sorted(iv):
        if ends and ends[0] <= s:
            heapq.heapreplace(ends, e)
        else:
            heapq.heappush(ends, e)
    return len(ends)


# ---------------- 8. Graph BFS / DFS ----------------
def num_islands(grid):                           # O(R*C), iterative to avoid recursion limits
    grid = [row[:] for row in grid]
    R, C, n = len(grid), len(grid[0]) if grid else 0, 0
    for i in range(R):
        for j in range(C):
            if grid[i][j] == "1":
                n += 1
                st = [(i, j)]
                grid[i][j] = "0"
                while st:
                    a, b = st.pop()
                    for x, y in ((a + 1, b), (a - 1, b), (a, b + 1), (a, b - 1)):
                        if 0 <= x < R and 0 <= y < C and grid[x][y] == "1":
                            grid[x][y] = "0"
                            st.append((x, y))
    return n


def shortest_path_grid(grid):                    # BFS, 8 directions, 0 = free
    n = len(grid)
    if grid[0][0] or grid[-1][-1]:
        return -1
    q, seen = deque([(0, 0, 1)]), {(0, 0)}
    while q:
        i, j, d = q.popleft()
        if (i, j) == (n - 1, n - 1):
            return d
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                x, y = i + di, j + dj
                if 0 <= x < n and 0 <= y < n and not grid[x][y] and (x, y) not in seen:
                    seen.add((x, y))
                    q.append((x, y, d + 1))
    return -1


# ---------------- 9. Topological sort ----------------
def course_order(n, prereqs):                    # Kahn's algorithm, O(V + E); [] if cycle
    g, indeg = defaultdict(list), [0] * n
    for a, b in prereqs:                         # b -> a
        g[b].append(a)
        indeg[a] += 1
    q = deque(i for i in range(n) if indeg[i] == 0)
    order = []
    while q:
        u = q.popleft()
        order.append(u)
        for v in g[u]:
            indeg[v] -= 1
            if indeg[v] == 0:
                q.append(v)
    return order if len(order) == n else []


def can_finish(n, prereqs):
    return len(course_order(n, prereqs)) == n


# ---------------- 10. Union-Find ----------------
class DSU:
    def __init__(self, n):
        self.p, self.r = list(range(n)), [0] * n

    def find(self, x):
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]        # path halving
            x = self.p[x]
        return x

    def union(self, a, b):
        a, b = self.find(a), self.find(b)
        if a == b:
            return False
        if self.r[a] < self.r[b]:
            a, b = b, a
        self.p[b] = a
        self.r[a] += self.r[a] == self.r[b]
        return True


def count_components(n, edges):
    d = DSU(n)
    return n - sum(d.union(a, b) for a, b in edges)


# ---------------- 11. Dijkstra ----------------
def network_delay(times, n, k):                  # O(E log V)
    g = defaultdict(list)
    for u, v, w in times:
        g[u].append((v, w))
    dist, h = {}, [(0, k)]
    while h:
        d, u = heapq.heappop(h)
        if u in dist:
            continue
        dist[u] = d
        for v, w in g[u]:
            if v not in dist:
                heapq.heappush(h, (d + w, v))
    return max(dist.values()) if len(dist) == n else -1


# ---------------- 12. Trie ----------------
class Trie:
    def __init__(self):
        self.root = {}

    def insert(self, w):
        node = self.root
        for c in w:
            node = node.setdefault(c, {})
        node["$"] = True

    def _walk(self, w):
        node = self.root
        for c in w:
            if c not in node:
                return None
            node = node[c]
        return node

    def search(self, w):
        n = self._walk(w)
        return bool(n and n.get("$"))

    def starts_with(self, p):
        return self._walk(p) is not None


# ---------------- 13. Backtracking ----------------
def subsets(nums):
    res = []

    def bt(i, cur):
        if i == len(nums):
            res.append(cur[:])
            return
        bt(i + 1, cur)
        cur.append(nums[i]); bt(i + 1, cur); cur.pop()
    bt(0, [])
    return res


def permutations(nums):
    res, used = [], [False] * len(nums)

    def bt(cur):
        if len(cur) == len(nums):
            res.append(cur[:])
            return
        for i, x in enumerate(nums):
            if not used[i]:
                used[i] = True; cur.append(x); bt(cur); cur.pop(); used[i] = False
    bt([])
    return res


def combination_sum(cands, target):
    cands, res = sorted(cands), []

    def bt(start, rem, cur):
        if rem == 0:
            res.append(cur[:])
            return
        for i in range(start, len(cands)):
            if cands[i] > rem:
                break
            cur.append(cands[i]); bt(i, rem - cands[i], cur); cur.pop()
    bt(0, target, [])
    return res


# ---------------- 14. DP ----------------
def coin_change(coins, amount):                  # O(amount * len(coins))
    INF = amount + 1
    dp = [0] + [INF] * amount
    for a in range(1, amount + 1):
        for c in coins:
            if c <= a:
                dp[a] = min(dp[a], dp[a - c] + 1)
    return dp[amount] if dp[amount] < INF else -1


def lis(nums):                                   # patience sorting, O(n log n)
    tails = []
    for x in nums:
        i = bisect.bisect_left(tails, x)
        if i == len(tails):
            tails.append(x)
        else:
            tails[i] = x
    return len(tails)


def edit_distance(a, b):                         # O(mn) time, O(n) space
    prev = list(range(len(b) + 1))
    for i in range(1, len(a) + 1):
        cur = [i] + [0] * len(b)
        for j in range(1, len(b) + 1):
            cur[j] = prev[j - 1] if a[i - 1] == b[j - 1] else 1 + min(prev[j - 1], prev[j], cur[j - 1])
        prev = cur
    return prev[-1]


def lcs(a, b):
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            dp[i][j] = dp[i - 1][j - 1] + 1 if a[i - 1] == b[j - 1] else max(dp[i - 1][j], dp[i][j - 1])
    return dp[-1][-1]


# ---------------- 15. Design: LRU cache ----------------
class _Node:
    __slots__ = ("k", "v", "prev", "next")

    def __init__(self, k=None, v=None):
        self.k, self.v, self.prev, self.next = k, v, None, None


class LRUCache:                                  # O(1) get/put; hashmap + doubly linked list
    def __init__(self, cap):
        self.cap, self.map = cap, {}
        self.head, self.tail = _Node(), _Node()  # sentinels; head.next = most recent
        self.head.next, self.tail.prev = self.tail, self.head

    def _remove(self, n):
        n.prev.next, n.next.prev = n.next, n.prev

    def _add_front(self, n):
        n.prev, n.next = self.head, self.head.next
        self.head.next.prev = n
        self.head.next = n

    def get(self, k):
        if k not in self.map:
            return -1
        n = self.map[k]
        self._remove(n); self._add_front(n)
        return n.v

    def put(self, k, v):
        if k in self.map:
            self._remove(self.map[k])
        n = self.map[k] = _Node(k, v)
        self._add_front(n)
        if len(self.map) > self.cap:
            lru = self.tail.prev
            self._remove(lru)
            del self.map[lru.k]


# ---------------- 16. ML-flavoured ----------------
def reservoir_sample(stream, k, rng=random):     # uniform k from a stream of unknown length, O(k) memory
    res = []
    for i, x in enumerate(stream):
        if i < k:
            res.append(x)
        else:
            j = rng.randint(0, i)
            if j < k:
                res[j] = x
    return res


class WeightedSampler:                           # LC 528: prefix sums + binary search
    def __init__(self, w):
        self.cum, s = [], 0
        for x in w:
            s += x
            self.cum.append(s)

    def pick(self, rng=random):
        return bisect.bisect_right(self.cum, rng.random() * self.cum[-1])


def sparse_matmul(A, B):                         # skip zeros: dict-of-rows representation
    Bn = [{j: v for j, v in enumerate(row) if v} for row in B]
    out = [[0] * len(B[0]) for _ in A]
    for i, row in enumerate(A):
        for k, a in enumerate(row):
            if a:
                for j, b in Bn[k].items():
                    out[i][j] += a * b
    return out


def kth_largest(nums, k, rng=random):            # quickselect, O(n) average
    nums, target = nums[:], len(nums) - k
    lo, hi = 0, len(nums) - 1
    while True:
        p = nums[rng.randint(lo, hi)]
        lt = [x for x in nums[lo:hi + 1] if x < p]
        eq = [x for x in nums[lo:hi + 1] if x == p]
        gt = [x for x in nums[lo:hi + 1] if x > p]
        nums[lo:hi + 1] = lt + eq + gt
        if target < lo + len(lt):
            hi = lo + len(lt) - 1
        elif target < lo + len(lt) + len(eq):
            return p
        else:
            lo = lo + len(lt) + len(eq)


# ---------------- tests ----------------
if __name__ == "__main__":
    random.seed(0)
    assert two_sum([2, 7, 11, 15], 9) == [0, 1]
    assert sorted(map(sorted, group_anagrams(["eat", "tea", "tan", "ate", "nat", "bat"]))) == [["ate", "eat", "tea"], ["bat"], ["nat", "tan"]]
    assert subarray_sum_k([1, 1, 1], 2) == 2 and subarray_sum_k([1, -1, 0], 0) == 3
    assert longest_unique_substring("abcabcbb") == 3 and longest_unique_substring("") == 0
    assert min_window("ADOBECODEBANC", "ABC") == "BANC" and min_window("a", "aa") == ""
    assert three_sum([-1, 0, 1, 2, -1, -4]) == [[-1, -1, 2], [-1, 0, 1]]
    assert max_area([1, 8, 6, 2, 5, 4, 8, 3, 7]) == 49
    assert search_rotated([4, 5, 6, 7, 0, 1, 2], 0) == 4 and search_rotated([4, 5, 6, 7, 0, 1, 2], 3) == -1
    assert min_eating_speed([3, 6, 7, 11], 8) == 4
    assert sorted(top_k_frequent([1, 1, 1, 2, 2, 3], 2)) == [1, 2]
    assert merge_k_sorted([[1, 4, 5], [1, 3, 4], [2, 6], []]) == [1, 1, 2, 3, 4, 4, 5, 6]
    mf = MedianFinder()
    for x, m in [(1, 1), (2, 1.5), (3, 2), (10, 2.5)]:
        mf.add(x); assert mf.median() == m
    assert daily_temperatures([73, 74, 75, 71, 69, 72, 76, 73]) == [1, 1, 4, 2, 1, 1, 0, 0]
    assert largest_rectangle([2, 1, 5, 6, 2, 3]) == 10
    assert merge_intervals([[1, 3], [2, 6], [8, 10], [15, 18]]) == [[1, 6], [8, 10], [15, 18]]
    assert min_meeting_rooms([[0, 30], [5, 10], [15, 20]]) == 2
    assert num_islands([list("11000"), list("11000"), list("00100"), list("00011")]) == 3
    assert shortest_path_grid([[0, 0, 0], [1, 1, 0], [1, 1, 0]]) == 4
    assert can_finish(2, [[1, 0]]) and not can_finish(2, [[1, 0], [0, 1]])
    o = course_order(4, [[1, 0], [2, 0], [3, 1], [3, 2]]); assert o[0] == 0 and o[-1] == 3
    assert count_components(5, [[0, 1], [1, 2], [3, 4]]) == 2
    assert network_delay([[2, 1, 1], [2, 3, 1], [3, 4, 1]], 4, 2) == 2
    t = Trie(); t.insert("apple")
    assert t.search("apple") and not t.search("app") and t.starts_with("app")
    assert len(subsets([1, 2, 3])) == 8 and len(permutations([1, 2, 3])) == 6
    assert combination_sum([2, 3, 6, 7], 7) == [[2, 2, 3], [7]]
    assert coin_change([1, 2, 5], 11) == 3 and coin_change([2], 3) == -1
    assert lis([10, 9, 2, 5, 3, 7, 101, 18]) == 4
    assert edit_distance("horse", "ros") == 3 and lcs("abcde", "ace") == 3
    c = LRUCache(2); c.put(1, 1); c.put(2, 2); assert c.get(1) == 1
    c.put(3, 3); assert c.get(2) == -1; c.put(4, 4); assert c.get(1) == -1 and c.get(3) == 3 and c.get(4) == 4

    counts = Counter()
    for _ in range(20000):
        counts.update(reservoir_sample(range(10), 3))
    assert all(abs(counts[i] / 20000 - 0.3) < 0.02 for i in range(10)), counts
    ws = WeightedSampler([1, 3]); picks = Counter(ws.pick() for _ in range(20000))
    assert abs(picks[1] / 20000 - 0.75) < 0.02
    A = [[1, 0, 0], [-1, 0, 3]]; B = [[7, 0, 0], [0, 0, 0], [0, 0, 1]]
    assert sparse_matmul(A, B) == [[7, 0, 0], [-7, 0, 3]]
    for _ in range(200):
        arr = [random.randint(-50, 50) for _ in range(random.randint(1, 30))]
        k = random.randint(1, len(arr))
        assert kth_largest(arr, k) == sorted(arr)[-k]
    print("all 16 patterns OK")
    print("OK")
