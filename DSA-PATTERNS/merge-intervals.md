## 🧠 Merge Intervals

The **Merge Intervals** pattern deals with problems involving overlapping intervals that need to be merged, inserted, or manipulated.

• Start by sorting intervals based on start time
• Iterate through sorted intervals and merge overlapping ones
• Two intervals overlap if start of one is ≤ end of another
• Useful for scheduling problems, calendar conflicts, and range operations

**Related:** [[Array]] [[Sorting]]

---

## 💡 Python Code Snippet

```python
def merge_intervals(intervals):
    """
    Merge overlapping intervals
    Example: [[1,3],[2,6],[8,10],[15,18]] → [[1,6],[8,10],[15,18]]
    """
    if not intervals:
        return []
    
    # Sort intervals by start time
    intervals.sort(key=lambda x: x[0])
    
    merged = [intervals[0]]
    
    for current in intervals[1:]:
        last_merged = merged[-1]
        
        # Check if current interval overlaps with last merged
        if current[0] <= last_merged[1]:
            # Overlapping: merge by extending end time
            merged[-1][1] = max(last_merged[1], current[1])
        else:
            # Non-overlapping: add as new interval
            merged.append(current)
    
    return merged

def insert_interval(intervals, new_interval):
    """
    Insert a new interval into sorted non-overlapping intervals
    """
    result = []
    i = 0
    n = len(intervals)
    
    # Add all intervals that end before new interval starts
    while i < n and intervals[i][1] < new_interval[0]:
        result.append(intervals[i])
        i += 1
    
    # Merge overlapping intervals with new interval
    while i < n and intervals[i][0] <= new_interval[1]:
        new_interval[0] = min(new_interval[0], intervals[i][0])
        new_interval[1] = max(new_interval[1], intervals[i][1])
        i += 1
    
    result.append(new_interval)
    
    # Add remaining intervals
    while i < n:
        result.append(intervals[i])
        i += 1
    
    return result

def can_attend_meetings(intervals):
    """
    Check if person can attend all meetings (no overlaps)
    """
    if not intervals:
        return True
    
    intervals.sort(key=lambda x: x[0])
    
    for i in range(1, len(intervals)):
        # If current meeting starts before previous one ends
        if intervals[i][0] < intervals[i-1][1]:
            return False
    
    return True
```

---

## 🔗 LeetCode Practice Problems (10)

- **Merge Intervals** – https://leetcode.com/problems/merge-intervals/
- **Insert Interval** – https://leetcode.com/problems/insert-interval/
- **Non-overlapping Intervals** – https://leetcode.com/problems/non-overlapping-intervals/
- **Meeting Rooms** – https://leetcode.com/problems/meeting-rooms/
- **Meeting Rooms II** – https://leetcode.com/problems/meeting-rooms-ii/
- **Interval List Intersections** – https://leetcode.com/problems/interval-list-intersections/
- **Employee Free Time** – https://leetcode.com/problems/employee-free-time/
- **My Calendar I** – https://leetcode.com/problems/my-calendar-i/
- **Minimum Number of Arrows to Burst Balloons** – https://leetcode.com/problems/minimum-number-of-arrows-to-burst-balloons/
- **Car Pooling** – https://leetcode.com/problems/car-pooling/

---

## 🧠 Flashcard (for spaced repetition)

What is the Merge Intervals pattern? :: • Handles problems with overlapping intervals by first sorting intervals by start time • Merges overlapping intervals where start of one ≤ end of another • Common for scheduling problems, calendar conflicts, and range operations [[merge-intervals]] 