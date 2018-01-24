import numpy as np

def add_to_queue(x, y, z, seen, queue, grid):
    if x < 0 or y < 0 or z < 0:
        return
    if x >= seen.shape[0] or y >= seen.shape[1] or z >= seen.shape[2]:
        return
    if seen[x, y, z]:
        return
    if grid[x, y, z]:
        queue.add((x, y, z))

def largest_contiguous(grid):
    curr_idx = 1
    counts = []
    seen = np.zeros_like(grid, dtype='uint32')
    result = np.zeros_like(grid, dtype='uint8')

    nx = grid.shape[0]
    ny = grid.shape[1]
    nz = grid.shape[2]

    for x in range(nx):
        for y in range(ny):
            for z in range(nz):
                if seen[x, y, z]:
                    continue

                if not grid[x, y, z]:
                    continue

                queue = set([( x, y, z )])
                counts.append(0)

                while queue:
                    _x, _y, _z = queue.pop()
                    if seen[_x, _y, _z]:
                        continue

                    seen[_x, _y, _z] = curr_idx
                    counts[curr_idx-1] += 1

                    add_to_queue(_x-1, _y, _z, seen, queue, grid)
                    add_to_queue(_x+1, _y, _z, seen, queue, grid)
                    add_to_queue(_x, _y-1, _z, seen, queue, grid)
                    add_to_queue(_x, _y+1, _z, seen, queue, grid)
                    add_to_queue(_x, _y, _z-1, seen, queue, grid)
                    add_to_queue(_x, _y, _z+1, seen, queue, grid)

                curr_idx += 1

    if counts:
        max_idx = counts.index(max(counts))+1
        result[seen == max_idx] = 1

    return result


if __name__ == '__main__':
    test = np.zeros([1, 1, 5])
    test[0, 0, :3] = 1
    test[0, 0, -1] = 1
    print(test)
    print(largest_contiguous(test))
