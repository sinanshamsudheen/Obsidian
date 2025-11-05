# Merge Intervals

## Explanation

The merge intervals pattern is used to combine overlapping intervals in a collection. It involves sorting intervals by start time and then merging overlapping ones. This pattern is particularly useful for scheduling problems, calendar applications, or when you need to find overlapping time periods in a dataset.

## Algorithm (Word-based)

- Sort intervals by start time
- Initialize result list with first interval
- Iterate through remaining intervals
- If current interval overlaps with last in result, merge them
- Otherwise, add current interval to result
- Return merged intervals

## Code Template

```python
def merge_intervals(intervals):
    """
    Merge overlapping intervals in a list
    """
    if not intervals:
        return []
    
    # Sort by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last = merged[-1]
        
        # Check if intervals overlap
        if current[0] <= last[1]:
            # Merge intervals by updating end time
            last[1] = max(last[1], current[1])
        else:
            # No overlap, add current interval
            merged.append(current)
    
    return merged
```

## Practice Questions

1. [Merge Intervals](https://leetcode.com/problems/merge-intervals/) - Medium
2. [Insert Interval](https://leetcode.com/problems/insert-interval/) - Medium
3. [Non-overlapping Intervals](https://leetcode.com/problems/non-overlapping-intervals/) - Medium
4. [Meeting Rooms](https://leetcode.com/problems/meeting-rooms/) - Easy
5. [Meeting Rooms II](https://leetcode.com/problems/meeting-rooms-ii/) - Medium
6. [Employee Free Time](https://leetcode.com/problems/employee-free-time/) - Hard
7. [Partition Labels](https://leetcode.com/problems/partition-labels/) - Medium
8. [Range Module](https://leetcode.com/problems/range-module/) - Hard
9. [My Calendar I](https://leetcode.com/problems/my-calendar-i/) - Medium
10. [Data Stream as Disjoint Intervals](https://leetcode.com/problems/data-stream-as-disjoint-intervals/) - Hard

## Notes

- Always sort intervals first by start time
- Overlap condition: current start <= last end
- Can be extended to find gaps between intervals

---

**Tags:** #dsa #merge-intervals #leetcode

**Last Reviewed:** 2025-11-05