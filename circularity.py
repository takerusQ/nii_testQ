def find_max_inscribed_circle(upper_edge, lower_edge, mask):
    height, width = mask.shape
    max_radius = 0
    center = None

    for x in range(width):
        for y in range(int(lower_edge[x]), int(upper_edge[x]) + 1):
            if y < 0 or y >= height:
                continue
            if not mask[y, x]:
                continue
            
            # Calculate the distance to the upper and lower edges
            dist_to_upper = upper_edge[x] - y
            dist_to_lower = y - lower_edge[x]
            
            # Find the minimum distance to the left and right edges
            left_edge = max(x - width, 0)
            right_edge = min(x + width, width - 1)
            
            if x > 0:
                dist_to_left = np.min([
                    np.min(np.abs(upper_edge[left_edge:x] - y)),
                    np.min(np.abs(lower_edge[left_edge:x] - y))
                ])
            else:
                dist_to_left = width
            
            if x < width - 1:
                dist_to_right = np.min([
                    np.min(np.abs(upper_edge[x+1:right_edge] - y)),
                    np.min(np.abs(lower_edge[x+1:right_edge] - y))
                ])
            else:
                dist_to_right = width
            
            # The radius is the minimum of all these distances
            radius = min(dist_to_upper, dist_to_lower, dist_to_left, dist_to_right)
            
            if radius > max_radius:
                max_radius = radius
                center = (x, y)

    return center, max_radius

# 他の関数は変更なし