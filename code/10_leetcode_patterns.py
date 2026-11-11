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


# =====================================================================
# 1. Hashing
# =====================================================================
def two_sum(nums, target):
    """LC 1. Return indices [i, j] such that nums[i] + nums[j] == target.

    Idea: while scanning, remember every value we've already seen and its index.
    For the current number, the partner we need is (target - current). If that
    partner is already in the hashmap, we're done.

    Time O(n), space O(n).
    """
    index_of_value = {}  # value -> index where we saw it

    for current_index, current_value in enumerate(nums):
        needed_partner = target - current_value

        # Did we already see a number that completes the pair?
        if needed_partner in index_of_value:
            partner_index = index_of_value[needed_partner]
            return [partner_index, current_index]

        # Store AFTER checking, so we never pair an element with itself.
        index_of_value[current_value] = current_index

    return []  # no valid pair


def group_anagrams(words):
    """LC 49. Group words that are anagrams of each other.

    Idea: two words are anagrams iff they have the same letter counts.
    Use the 26-length letter-count tuple as a hashmap key.
    (Alternative key: ''.join(sorted(word)), which costs O(k log k) per word.)

    Time O(n * k) where k = average word length, space O(n * k).
    """
    groups = defaultdict(list)  # letter-count signature -> list of words

    for word in words:
        letter_counts = [0] * 26
        for char in word:
            letter_index = ord(char) - ord("a")
            letter_counts[letter_index] += 1

        # Lists are not hashable, so convert to a tuple to use as a dict key.
        signature = tuple(letter_counts)
        groups[signature].append(word)

    return list(groups.values())


def subarray_sum_k(nums, k):
    """LC 560. Count contiguous subarrays whose sum equals k.

    Idea (prefix sum + hashmap):
        sum(nums[i+1..j]) = prefix[j] - prefix[i]
    So a subarray ending at j sums to k  <=>  some earlier prefix == prefix[j] - k.
    Keep a count of how many times each prefix sum has appeared so far.
    Works with negative numbers (sliding window would not).

    Time O(n), space O(n).
    """
    # The empty prefix (before any element) has sum 0 and appears once.
    # This lets subarrays that start at index 0 be counted.
    prefix_sum_count = Counter()
    prefix_sum_count[0] = 1

    running_sum = 0
    subarray_count = 0

    for value in nums:
        running_sum += value

        # If (running_sum - k) was seen before, each occurrence marks a start
        # point where the subarray ending here sums to exactly k.
        needed_prefix = running_sum - k
        subarray_count += prefix_sum_count[needed_prefix]

        # Record the current prefix so later positions can use it.
        prefix_sum_count[running_sum] += 1

    return subarray_count


# =====================================================================
# 2. Sliding window
# =====================================================================
def longest_unique_substring(s):
    """LC 3. Length of the longest substring without repeating characters.

    Idea: sliding window [left, right]. Invariant: the window contains only
    unique characters. When s[right] already appears inside the window, jump
    `left` to just after its previous position.

    Time O(n), space O(min(n, alphabet size)).
    """
    last_seen_index = {}  # char -> most recent index where it appeared
    left = 0
    best_length = 0

    for right, char in enumerate(s):
        # Only a duplicate if its previous occurrence is INSIDE the current window.
        if char in last_seen_index and last_seen_index[char] >= left:
            left = last_seen_index[char] + 1  # shrink window past the duplicate

        last_seen_index[char] = right

        window_length = right - left + 1
        best_length = max(best_length, window_length)

    return best_length


def min_window(s, t):
    """LC 76. Smallest substring of s that contains every char of t (with multiplicity).

    Idea: sliding window.
      1. Expand `right` until the window covers all of t (chars_missing == 0).
      2. Then shrink `left` as much as possible while it still covers t,
         recording the best (shortest) window along the way.
    need[c] = how many more copies of c the window still needs (can go negative
    when the window has extras).

    Time O(|s| + |t|), space O(|t|).
    """
    need = Counter(t)
    chars_missing = len(t)  # total chars of t not yet covered by the window

    left = 0
    best_start = 0
    best_length = float("inf")

    for right, char in enumerate(s):
        # --- expand: add s[right] to the window ---
        if need[char] > 0:
            chars_missing -= 1  # this char was actually useful
        need[char] -= 1

        # --- shrink: window is valid, try to make it smaller ---
        while chars_missing == 0:
            window_length = right - left + 1
            if window_length < best_length:
                best_start = left
                best_length = window_length

            # Remove s[left] from the window.
            left_char = s[left]
            need[left_char] += 1
            if need[left_char] > 0:
                chars_missing += 1  # we just dropped a required char -> window invalid
            left += 1

    if best_length == float("inf"):
        return ""
    return s[best_start:best_start + best_length]


# =====================================================================
# 3. Two pointers
# =====================================================================
def three_sum(nums):
    """LC 15. All unique triplets [a, b, c] with a + b + c == 0.

    Idea: sort, fix the first element nums[i], then find pairs in the rest with
    two pointers (left from the start, right from the end). Skip duplicate
    values to avoid repeated triplets.

    Time O(n^2), space O(1) extra (ignoring output and the sort).
    """
    sorted_nums = sorted(nums)
    n = len(sorted_nums)
    triplets = []

    for i in range(n - 2):
        # Skip duplicate first elements: same value would give the same triplets.
        if i > 0 and sorted_nums[i] == sorted_nums[i - 1]:
            continue

        left = i + 1
        right = n - 1

        while left < right:
            total = sorted_nums[i] + sorted_nums[left] + sorted_nums[right]

            if total < 0:
                left += 1  # need a bigger sum -> move left pointer right
            elif total > 0:
                right -= 1  # need a smaller sum -> move right pointer left
            else:
                triplets.append([sorted_nums[i], sorted_nums[left], sorted_nums[right]])

                # Skip duplicate second elements.
                while left < right and sorted_nums[left] == sorted_nums[left + 1]:
                    left += 1

                left += 1
                right -= 1

    return triplets


def max_area(h):
    """LC 11. Container with most water: pick two lines maximising width * min height.

    Idea: start with the widest container (both ends). The area is limited by
    the SHORTER line, so moving the taller one inward can never help — always
    move the shorter side.

    Time O(n), space O(1).
    """
    heights = h
    left = 0
    right = len(heights) - 1
    best_area = 0

    while left < right:
        width = right - left
        water_height = min(heights[left], heights[right])
        area = width * water_height
        best_area = max(best_area, area)

        # Move the shorter wall: it's the bottleneck.
        if heights[left] < heights[right]:
            left += 1
        else:
            right -= 1

    return best_area


# =====================================================================
# 4. Binary search
# =====================================================================
def search_rotated(nums, target):
    """LC 33. Find target's index in a rotated sorted array (distinct values), or -1.

    Idea: at any mid, at least one half [lo..mid] or [mid..hi] is sorted.
    Check whether target lies inside the sorted half; if yes search there,
    otherwise search the other half.

    Time O(log n), space O(1).
    """
    lo = 0
    hi = len(nums) - 1

    while lo <= hi:
        mid = (lo + hi) // 2

        if nums[mid] == target:
            return mid

        left_half_is_sorted = nums[lo] <= nums[mid]

        if left_half_is_sorted:
            target_in_left_half = nums[lo] <= target < nums[mid]
            if target_in_left_half:
                hi = mid - 1
            else:
                lo = mid + 1
        else:
            # Then the right half [mid..hi] must be sorted.
            target_in_right_half = nums[mid] < target <= nums[hi]
            if target_in_right_half:
                lo = mid + 1
            else:
                hi = mid - 1

    return -1


def min_eating_speed(piles, h):
    """LC 875. Koko eats bananas: smallest speed k so all piles finish within h hours.

    Idea: "binary search on the answer". If speed k works, any speed > k also
    works (monotonic). So binary search the smallest feasible k in [1, max(piles)].

    Time O(n log max(piles)), space O(1).
    """

    def hours_needed(speed):
        total_hours = 0
        for pile in piles:
            # ceil(pile / speed) without floats
            total_hours += (pile + speed - 1) // speed
        return total_hours

    lo = 1
    hi = max(piles)  # at this speed every pile takes exactly 1 hour

    # Invariant: the answer is always within [lo, hi].
    while lo < hi:
        mid = (lo + hi) // 2
        if hours_needed(mid) <= h:
            hi = mid  # mid works; maybe something smaller works too
        else:
            lo = mid + 1  # mid too slow; answer must be bigger

    return lo


# =====================================================================
# 5. Heap
# =====================================================================
def top_k_frequent(nums, k):
    """LC 347. Return the k most frequent elements.

    Idea: count with a hashmap, then keep a MIN-heap of size k keyed by
    frequency. Whenever the heap grows beyond k, pop the least frequent.
    What's left are the k most frequent.
    (Alternative: bucket sort by frequency for O(n).)

    Time O(n log k), space O(n).
    """
    frequency = Counter(nums)

    min_heap = []  # (count, value); smallest count on top
    for value, count in frequency.items():
        heapq.heappush(min_heap, (count, value))
        if len(min_heap) > k:
            heapq.heappop(min_heap)  # drop the least frequent so far

    result = []
    for count, value in min_heap:
        result.append(value)
    return result


def merge_k_sorted(lists):
    """LC 23. Merge k sorted lists into one sorted list.

    Idea: a min-heap holds the current front element of each list. Pop the
    smallest, append it to the output, and push the next element from the
    same list.

    Time O(N log k) where N = total elements, space O(k) for the heap.
    """
    # Heap entries: (value, which_list, index_in_that_list).
    # which_list also breaks ties so we never compare beyond integers.
    min_heap = []
    for list_index, current_list in enumerate(lists):
        if current_list:  # skip empty lists
            min_heap.append((current_list[0], list_index, 0))
    heapq.heapify(min_heap)

    merged = []
    while min_heap:
        value, list_index, element_index = heapq.heappop(min_heap)
        merged.append(value)

        next_index = element_index + 1
        source_list = lists[list_index]
        if next_index < len(source_list):
            heapq.heappush(min_heap, (source_list[next_index], list_index, next_index))

    return merged


class MedianFinder:
    """LC 295. Stream median.

    Idea: two heaps.
      - small_half: MAX-heap with the smaller half (stored negated, since
        heapq is a min-heap).
      - large_half: MIN-heap with the larger half.
    Invariants: every element in small_half <= every element in large_half,
    and len(small_half) == len(large_half) or len(large_half) + 1.

    add: O(log n), median: O(1).
    """

    def __init__(self):
        self.small_half = []  # max-heap via negation
        self.large_half = []  # min-heap

    def add(self, x):
        # Step 1: push into small half.
        heapq.heappush(self.small_half, -x)

        # Step 2: move small half's largest into large half, which keeps the
        # ordering invariant (everything in small <= everything in large).
        largest_of_small = -heapq.heappop(self.small_half)
        heapq.heappush(self.large_half, largest_of_small)

        # Step 3: rebalance sizes so small half is never smaller than large half.
        if len(self.large_half) > len(self.small_half):
            smallest_of_large = heapq.heappop(self.large_half)
            heapq.heappush(self.small_half, -smallest_of_large)

    def median(self):
        max_of_small = -self.small_half[0]

        # Odd count: small half holds the extra element, which is the median.
        if len(self.small_half) > len(self.large_half):
            return max_of_small

        # Even count: average of the two middle elements.
        min_of_large = self.large_half[0]
        return (max_of_small + min_of_large) / 2


# =====================================================================
# 6. Monotonic stack
# =====================================================================
def daily_temperatures(t):
    """LC 739. For each day, how many days until a warmer temperature (0 if never).

    Idea: monotonic DECREASING stack of indices still waiting for a warmer day.
    When today is warmer than the day on top of the stack, that day's answer
    is found: pop it and record the distance.

    Time O(n) (each index pushed/popped once), space O(n).
    """
    temperatures = t
    days_to_wait = [0] * len(temperatures)
    waiting_stack = []  # indices; their temperatures are decreasing bottom -> top

    for today, today_temp in enumerate(temperatures):
        # Today resolves every colder day on top of the stack.
        while waiting_stack and temperatures[waiting_stack[-1]] < today_temp:
            colder_day = waiting_stack.pop()
            days_to_wait[colder_day] = today - colder_day

        waiting_stack.append(today)

    # Anything left in the stack never sees a warmer day -> stays 0.
    return days_to_wait


def largest_rectangle(heights):
    """LC 84. Largest rectangle area in a histogram.

    Idea: monotonic INCREASING stack of (start_index, height). When a shorter
    bar arrives, every taller bar on the stack can't extend further right, so
    pop it and compute its area. The new bar can extend LEFT to where the
    popped bar started.
    A sentinel height 0 at the end flushes the stack.

    Time O(n), space O(n).
    """
    stack = []  # (start_index, height); heights increasing bottom -> top
    best_area = 0
    heights_with_sentinel = heights + [0]

    for index, height in enumerate(heights_with_sentinel):
        start_index = index

        # Pop taller (or equal) bars: their rectangle ends right before `index`.
        while stack and stack[-1][1] >= height:
            popped_start, popped_height = stack.pop()
            width = index - popped_start
            best_area = max(best_area, popped_height * width)

            # The current, shorter bar can stretch back to where the popped one began.
            start_index = popped_start

        stack.append((start_index, height))

    return best_area


# =====================================================================
# 7. Intervals
# =====================================================================
def merge_intervals(iv):
    """LC 56. Merge all overlapping intervals.

    Idea: sort by start. Then an interval overlaps the previous merged one iff
    its start <= the last merged end. If so, extend that end; otherwise start a
    new merged interval.

    Time O(n log n), space O(n).
    """
    intervals = sorted(iv)  # sorts by start, then end
    merged = []

    for start, end in intervals:
        if merged and start <= merged[-1][1]:
            # Overlaps the last merged interval -> extend it.
            last_end = merged[-1][1]
            merged[-1][1] = max(last_end, end)
        else:
            # No overlap -> new interval.
            merged.append([start, end])

    return merged


def min_meeting_rooms(iv):
    """LC 253. Minimum number of meeting rooms needed.

    Idea: process meetings by start time. Keep a min-heap of END times of
    meetings currently using a room. If the earliest-ending room is free by
    the time this meeting starts, reuse it; otherwise open a new room.
    The heap size at the end = rooms needed.

    Time O(n log n), space O(n).
    """
    meetings = sorted(iv)
    room_end_times = []  # min-heap: when each occupied room frees up

    for start, end in meetings:
        earliest_free_room_available = room_end_times and room_end_times[0] <= start

        if earliest_free_room_available:
            # Reuse that room: replace its end time with this meeting's end.
            heapq.heapreplace(room_end_times, end)
        else:
            # All rooms busy: need a new one.
            heapq.heappush(room_end_times, end)

    return len(room_end_times)


# =====================================================================
# 8. Graph BFS / DFS
# =====================================================================
def num_islands(grid):
    """LC 200. Count islands of '1's (4-directionally connected) in a grid.

    Idea: scan every cell. When we find unvisited land, that's a new island:
    flood-fill (DFS) to mark all connected land as visited ("0").
    Iterative stack to avoid Python's recursion limit on big grids.

    Time O(R * C), space O(R * C).
    """
    # Copy so we don't mutate the caller's grid.
    grid = [row[:] for row in grid]
    if not grid:
        return 0

    num_rows = len(grid)
    num_cols = len(grid[0])
    island_count = 0
    directions = [(1, 0), (-1, 0), (0, 1), (0, -1)]

    for row in range(num_rows):
        for col in range(num_cols):
            if grid[row][col] != "1":
                continue

            # Found a new island -> sink all of it.
            island_count += 1
            grid[row][col] = "0"  # mark visited when pushing
            stack = [(row, col)]

            while stack:
                cur_row, cur_col = stack.pop()
                for d_row, d_col in directions:
                    next_row = cur_row + d_row
                    next_col = cur_col + d_col
                    inside_grid = 0 <= next_row < num_rows and 0 <= next_col < num_cols
                    if inside_grid and grid[next_row][next_col] == "1":
                        grid[next_row][next_col] = "0"
                        stack.append((next_row, next_col))

    return island_count


def shortest_path_grid(grid):
    """LC 1091. Shortest clear path (0 = free, 1 = blocked) from top-left to
    bottom-right in an n x n grid, moving in 8 directions. Path length counts cells.

    Idea: BFS from the start; BFS explores in order of distance, so the first
    time we reach the target is the shortest path.

    Time O(n^2), space O(n^2).
    """
    n = len(grid)
    start_blocked = grid[0][0] == 1
    end_blocked = grid[n - 1][n - 1] == 1
    if start_blocked or end_blocked:
        return -1

    target = (n - 1, n - 1)
    queue = deque()
    queue.append((0, 0, 1))  # (row, col, path_length_so_far)
    visited = {(0, 0)}

    while queue:
        row, col, path_length = queue.popleft()

        if (row, col) == target:
            return path_length

        # All 8 neighbours (d_row, d_col) in {-1,0,1}^2 except (0,0).
        for d_row in (-1, 0, 1):
            for d_col in (-1, 0, 1):
                if d_row == 0 and d_col == 0:
                    continue
                next_row = row + d_row
                next_col = col + d_col

                if not (0 <= next_row < n and 0 <= next_col < n):
                    continue
                if grid[next_row][next_col] == 1:
                    continue
                if (next_row, next_col) in visited:
                    continue

                # Mark visited when ENQUEUING, so each cell is queued at most once.
                visited.add((next_row, next_col))
                queue.append((next_row, next_col, path_length + 1))

    return -1


# =====================================================================
# 9. Topological sort
# =====================================================================
def course_order(n, prereqs):
    """LC 210. Return an order to take n courses given [course, prerequisite]
    pairs; return [] if impossible (there is a cycle).

    Idea: Kahn's algorithm (BFS topological sort). Repeatedly take a course
    with no remaining prerequisites (in-degree 0), then "remove" it, lowering
    the in-degree of courses that depend on it.
    If some courses are never taken, they're stuck in a cycle.

    Time O(V + E), space O(V + E).
    """
    dependents = defaultdict(list)  # prerequisite -> courses that need it
    in_degree = [0] * n  # number of unmet prerequisites per course

    for course, prerequisite in prereqs:
        # Edge: prerequisite -> course
        dependents[prerequisite].append(course)
        in_degree[course] += 1

    # Start with courses that have no prerequisites.
    ready_queue = deque()
    for course in range(n):
        if in_degree[course] == 0:
            ready_queue.append(course)

    order = []
    while ready_queue:
        course = ready_queue.popleft()
        order.append(course)

        # Taking this course satisfies one prerequisite for each dependent.
        for next_course in dependents[course]:
            in_degree[next_course] -= 1
            if in_degree[next_course] == 0:
                ready_queue.append(next_course)

    all_courses_taken = len(order) == n
    if all_courses_taken:
        return order
    return []  # cycle detected


def can_finish(n, prereqs):
    """LC 207. Can all courses be finished? True iff the prerequisite graph has
    no cycle, i.e. a full topological order exists. Time O(V + E).
    """
    order = course_order(n, prereqs)
    return len(order) == n


# =====================================================================
# 10. Union-Find
# =====================================================================
class DSU:
    """Disjoint Set Union (Union-Find) with path compression + union by rank.

    find / union are amortised ~O(1) (inverse Ackermann).
    """

    def __init__(self, n):
        self.parent = list(range(n))  # initially every node is its own root
        self.rank = [0] * n  # upper bound on tree height

    def find(self, x):
        """Return the root of x's set."""
        while self.parent[x] != x:
            # Path halving: point x to its grandparent, flattening the tree.
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        """Merge the sets of a and b. Return True if they were separate."""
        root_a = self.find(a)
        root_b = self.find(b)

        if root_a == root_b:
            return False  # already in the same set

        # Attach the shorter tree under the taller one.
        if self.rank[root_a] < self.rank[root_b]:
            root_a, root_b = root_b, root_a
        self.parent[root_b] = root_a

        # Height only grows when both trees had equal rank.
        if self.rank[root_a] == self.rank[root_b]:
            self.rank[root_a] += 1

        return True


def count_components(n, edges):
    """LC 323. Number of connected components in an undirected graph.

    Idea: start with n components; every edge that joins two DIFFERENT
    components reduces the count by one.

    Time O(E * alpha(n)), space O(n).
    """
    dsu = DSU(n)
    components = n

    for a, b in edges:
        merged_two_components = dsu.union(a, b)
        if merged_two_components:
            components -= 1

    return components


# =====================================================================
# 11. Dijkstra
# =====================================================================
def network_delay(times, n, k):
    """LC 743. Signal from node k: time until all n nodes receive it, or -1.
    times = [u, v, w] directed edges with travel time w.

    Idea: Dijkstra with a min-heap. The first time a node is popped, its
    distance is final (all weights non-negative). Answer = the largest
    shortest distance.

    Time O(E log V), space O(V + E).
    """
    graph = defaultdict(list)  # node -> [(neighbour, weight)]
    for source, destination, weight in times:
        graph[source].append((destination, weight))

    shortest_distance = {}  # finalised distances
    min_heap = [(0, k)]  # (distance from k, node)

    while min_heap:
        distance, node = heapq.heappop(min_heap)

        # A node can be pushed several times; only the first pop matters.
        if node in shortest_distance:
            continue
        shortest_distance[node] = distance

        for neighbour, weight in graph[node]:
            if neighbour not in shortest_distance:
                heapq.heappush(min_heap, (distance + weight, neighbour))

    every_node_reached = len(shortest_distance) == n
    if not every_node_reached:
        return -1
    return max(shortest_distance.values())


# =====================================================================
# 12. Trie
# =====================================================================
class Trie:
    """LC 208. Prefix tree supporting insert / search / starts_with.

    Each node is a dict: char -> child node. The special key "$" marks that a
    complete word ends at this node.

    All operations O(L) where L = word length.
    """

    END_OF_WORD = "$"

    def __init__(self):
        self.root = {}

    def insert(self, w):
        node = self.root
        for char in w:
            if char not in node:
                node[char] = {}  # create the missing child
            node = node[char]
        node[self.END_OF_WORD] = True

    def _walk(self, w):
        """Follow w's characters from the root; return the final node or None."""
        node = self.root
        for char in w:
            if char not in node:
                return None
            node = node[char]
        return node

    def search(self, w):
        """True only if w was inserted as a whole word (not just a prefix)."""
        node = self._walk(w)
        if node is None:
            return False
        return node.get(self.END_OF_WORD, False)

    def starts_with(self, p):
        """True if any inserted word starts with prefix p."""
        node = self._walk(p)
        return node is not None


# =====================================================================
# 13. Backtracking
# =====================================================================
def subsets(nums):
    """LC 78. All subsets (power set) of distinct nums.

    Idea: for each index, make a binary choice — skip nums[i] or take it.
    Recurse until we've decided for every element.

    Time O(n * 2^n), space O(n) recursion depth (plus output).
    """
    all_subsets = []
    current_subset = []

    def backtrack(index):
        # Decided for every element -> record a copy of the current subset.
        if index == len(nums):
            all_subsets.append(current_subset[:])
            return

        # Choice 1: skip nums[index].
        backtrack(index + 1)

        # Choice 2: take nums[index], recurse, then undo the choice.
        current_subset.append(nums[index])
        backtrack(index + 1)
        current_subset.pop()

    backtrack(0)
    return all_subsets


def permutations(nums):
    """LC 46. All permutations of distinct nums.

    Idea: build the permutation one position at a time; at each step try every
    element not yet used. `used[i]` tracks which elements are already placed.

    Time O(n * n!), space O(n) recursion depth (plus output).
    """
    all_permutations = []
    current_permutation = []
    used = [False] * len(nums)

    def backtrack():
        if len(current_permutation) == len(nums):
            all_permutations.append(current_permutation[:])
            return

        for i in range(len(nums)):
            if used[i]:
                continue

            # Choose
            used[i] = True
            current_permutation.append(nums[i])

            # Explore
            backtrack()

            # Un-choose (restore state for the next candidate)
            current_permutation.pop()
            used[i] = False

    backtrack()
    return all_permutations


def combination_sum(cands, target):
    """LC 39. All unique combinations of candidates summing to target; each
    candidate may be reused any number of times.

    Idea: backtracking. To avoid duplicates like [2,3] and [3,2], only pick
    candidates at index >= start (non-decreasing order). Passing `i` (not i+1)
    allows reuse. Sorting lets us stop early once a candidate exceeds what's left.

    Time exponential (roughly O(N^(T/min))), space O(T/min) recursion depth.
    """
    candidates = sorted(cands)
    all_combinations = []
    current_combination = []

    def backtrack(start_index, remaining):
        if remaining == 0:
            all_combinations.append(current_combination[:])
            return

        for i in range(start_index, len(candidates)):
            candidate = candidates[i]

            # Sorted: if this one is too big, all later ones are too.
            if candidate > remaining:
                break

            current_combination.append(candidate)
            backtrack(i, remaining - candidate)  # i, not i+1: reuse allowed
            current_combination.pop()

    backtrack(0, target)
    return all_combinations


# =====================================================================
# 14. Dynamic programming
# =====================================================================
def coin_change(coins, amount):
    """LC 322. Fewest coins to make `amount` (unlimited coins); -1 if impossible.

    DP: min_coins[a] = fewest coins summing to a.
        min_coins[0] = 0
        min_coins[a] = 1 + min(min_coins[a - coin]) over coins <= a

    Time O(amount * len(coins)), space O(amount).
    """
    IMPOSSIBLE = amount + 1  # more coins than could ever be needed (acts as infinity)

    min_coins = [IMPOSSIBLE] * (amount + 1)
    min_coins[0] = 0

    for current_amount in range(1, amount + 1):
        for coin in coins:
            if coin > current_amount:
                continue
            # Use this coin last: 1 coin + best way to make the rest.
            candidate = min_coins[current_amount - coin] + 1
            if candidate < min_coins[current_amount]:
                min_coins[current_amount] = candidate

    if min_coins[amount] == IMPOSSIBLE:
        return -1
    return min_coins[amount]


def lis(nums):
    """LC 300. Length of the longest strictly increasing subsequence.

    Idea (patience sorting): tails[i] = smallest possible tail value of an
    increasing subsequence of length i+1. `tails` stays sorted, so for each
    number we binary-search where it goes:
      - larger than every tail -> extends the longest subsequence (append)
      - otherwise -> replace the first tail >= it (a smaller tail is better)
    len(tails) is the answer (tails itself is NOT necessarily a valid LIS).

    Time O(n log n), space O(n).
    """
    tails = []

    for number in nums:
        position = bisect.bisect_left(tails, number)  # first tail >= number

        if position == len(tails):
            tails.append(number)
        else:
            tails[position] = number

    return len(tails)


def edit_distance(a, b):
    """LC 72. Minimum insert/delete/replace operations to turn a into b.

    DP: dist[i][j] = edit distance between a[:i] and b[:j].
      - if a[i-1] == b[j-1]: dist[i][j] = dist[i-1][j-1]          (no cost)
      - else: 1 + min(dist[i-1][j-1],   # replace
                      dist[i-1][j],     # delete from a
                      dist[i][j-1])     # insert into a
    Only the previous row is needed, so keep two rows.

    Time O(m * n), space O(n).
    """
    len_a = len(a)
    len_b = len(b)

    # Row for i = 0: turning "" into b[:j] takes j insertions.
    previous_row = list(range(len_b + 1))

    for i in range(1, len_a + 1):
        current_row = [0] * (len_b + 1)
        current_row[0] = i  # turning a[:i] into "" takes i deletions

        for j in range(1, len_b + 1):
            if a[i - 1] == b[j - 1]:
                current_row[j] = previous_row[j - 1]
            else:
                replace_cost = previous_row[j - 1]
                delete_cost = previous_row[j]
                insert_cost = current_row[j - 1]
                current_row[j] = 1 + min(replace_cost, delete_cost, insert_cost)

        previous_row = current_row

    return previous_row[len_b]


def lcs(a, b):
    """LC 1143. Length of the longest common subsequence of a and b.

    DP: common[i][j] = LCS length of a[:i] and b[:j].
      - if a[i-1] == b[j-1]: common[i][j] = common[i-1][j-1] + 1
      - else:                common[i][j] = max(common[i-1][j], common[i][j-1])

    Time O(m * n), space O(m * n).
    """
    len_a = len(a)
    len_b = len(b)

    # Extra row/column of zeros for empty prefixes.
    common = []
    for _ in range(len_a + 1):
        common.append([0] * (len_b + 1))

    for i in range(1, len_a + 1):
        for j in range(1, len_b + 1):
            if a[i - 1] == b[j - 1]:
                # Matching chars extend the LCS of both shorter prefixes.
                common[i][j] = common[i - 1][j - 1] + 1
            else:
                # Drop the last char of a or of b, whichever is better.
                common[i][j] = max(common[i - 1][j], common[i][j - 1])

    return common[len_a][len_b]


# =====================================================================
# 15. Design: LRU cache
# =====================================================================
class _Node:
    """Doubly linked list node used by LRUCache."""

    __slots__ = ("k", "v", "prev", "next")

    def __init__(self, k=None, v=None):
        self.k = k  # key (needed to delete from the hashmap on eviction)
        self.v = v  # value
        self.prev = None
        self.next = None


class LRUCache:
    """LC 146. Least-Recently-Used cache with O(1) get and put.

    Idea: hashmap (key -> node) for O(1) lookup + doubly linked list for O(1)
    reordering. List order = recency: right after `head` is the most recently
    used, right before `tail` is the least recently used.
    `head` and `tail` are dummy sentinels so we never special-case empty lists.
    (In Python, collections.OrderedDict + move_to_end also works.)
    """

    def __init__(self, cap):
        self.capacity = cap
        self.node_of_key = {}

        self.head = _Node()  # sentinel: head.next = most recently used
        self.tail = _Node()  # sentinel: tail.prev = least recently used
        self.head.next = self.tail
        self.tail.prev = self.head

    def _remove(self, node):
        """Unlink node from the list."""
        previous_node = node.prev
        next_node = node.next
        previous_node.next = next_node
        next_node.prev = previous_node

    def _add_front(self, node):
        """Insert node right after head (= mark as most recently used)."""
        first_real_node = self.head.next

        node.prev = self.head
        node.next = first_real_node

        first_real_node.prev = node
        self.head.next = node

    def get(self, k):
        if k not in self.node_of_key:
            return -1

        node = self.node_of_key[k]

        # Accessing a key makes it the most recently used.
        self._remove(node)
        self._add_front(node)

        return node.v

    def put(self, k, v):
        # If the key exists, drop the old node; we'll insert a fresh one.
        if k in self.node_of_key:
            old_node = self.node_of_key[k]
            self._remove(old_node)

        new_node = _Node(k, v)
        self.node_of_key[k] = new_node
        self._add_front(new_node)

        # Over capacity -> evict the least recently used (just before tail).
        if len(self.node_of_key) > self.capacity:
            least_recent = self.tail.prev
            self._remove(least_recent)
            del self.node_of_key[least_recent.k]


# =====================================================================
# 16. ML-flavoured
# =====================================================================
def reservoir_sample(stream, k, rng=random):
    """Pick k items uniformly at random from a stream of unknown length,
    using O(k) memory (one pass).

    Idea (Algorithm R): keep the first k items. For the i-th item (0-indexed,
    i >= k), keep it with probability k / (i + 1) by drawing j in [0, i]; if
    j < k, it replaces reservoir[j]. By induction every item ends up in the
    reservoir with probability k / n.

    Time O(n), space O(k).
    """
    reservoir = []

    for index, item in enumerate(stream):
        if index < k:
            # Fill the reservoir first.
            reservoir.append(item)
            continue

        random_slot = rng.randint(0, index)  # inclusive on both ends
        if random_slot < k:
            # Happens with probability k / (index + 1).
            reservoir[random_slot] = item

    return reservoir


class WeightedSampler:
    """LC 528. Random pick with weight: return index i with probability
    w[i] / sum(w).

    Idea: build prefix sums; this splits [0, total) into segments whose lengths
    are the weights. Draw a uniform number in [0, total) and binary-search which
    segment it lands in.

    __init__ O(n), pick O(log n).
    """

    def __init__(self, w):
        self.cum = []  # cum[i] = w[0] + ... + w[i]
        running_total = 0
        for weight in w:
            running_total += weight
            self.cum.append(running_total)

    def pick(self, rng=random):
        total_weight = self.cum[-1]
        point = rng.random() * total_weight  # uniform in [0, total_weight)

        # Index i such that cum[i-1] <= point < cum[i].
        return bisect.bisect_right(self.cum, point)


def sparse_matmul(A, B):
    """LC 311. Multiply matrices A (m x k) and B (k x n) that are mostly zeros.

    Idea: only non-zero entries contribute. Pre-store each row of B as a dict
    {column: value} of non-zeros, then for every non-zero A[i][k] add
    A[i][k] * B[k][j] to out[i][j] for the non-zero B[k][j].

    Time O(nnz(A) * avg nnz per B row), space O(nnz(B) + m * n).
    """
    num_rows_out = len(A)
    num_cols_out = len(B[0])

    # Compress B: for each row, keep only non-zero (column -> value) entries.
    nonzero_rows_of_B = []
    for row in B:
        nonzero_entries = {}
        for col, value in enumerate(row):
            if value != 0:
                nonzero_entries[col] = value
        nonzero_rows_of_B.append(nonzero_entries)

    result = []
    for _ in range(num_rows_out):
        result.append([0] * num_cols_out)

    for i in range(num_rows_out):
        for shared_index, a_value in enumerate(A[i]):
            if a_value == 0:
                continue  # contributes nothing
            for j, b_value in nonzero_rows_of_B[shared_index].items():
                result[i][j] += a_value * b_value

    return result


def kth_largest(nums, k, rng=random):
    """LC 215. k-th largest element in an unsorted array.

    Idea: quickselect. The k-th largest is the element at index (n - k) once
    sorted ascending. Pick a random pivot, partition the current range into
    < pivot | == pivot | > pivot, then continue only in the part that
    contains the target index.
    (Alternative: min-heap of size k, O(n log k).)

    Time O(n) average, O(n^2) worst case; space O(n) here (3-way partition lists).
    """
    values = nums[:]  # don't mutate the caller's list
    target_index = len(values) - k  # position in ascending sorted order

    lo = 0
    hi = len(values) - 1

    # Invariant: the target index always lies within [lo, hi].
    while True:
        pivot = values[rng.randint(lo, hi)]

        current_range = values[lo:hi + 1]
        smaller = []
        equal = []
        larger = []
        for value in current_range:
            if value < pivot:
                smaller.append(value)
            elif value == pivot:
                equal.append(value)
            else:
                larger.append(value)

        # Write the partitioned range back in place.
        values[lo:hi + 1] = smaller + equal + larger

        equal_start = lo + len(smaller)  # first index holding the pivot value
        larger_start = equal_start + len(equal)  # first index after the pivots

        if target_index < equal_start:
            hi = equal_start - 1  # answer is in the "smaller" part
        elif target_index < larger_start:
            return pivot  # answer is the pivot itself
        else:
            lo = larger_start  # answer is in the "larger" part


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
