def part2Test(data):
    labels = [i for i in range(len(data))]

    while len(clusters) > 1:
        clusters = findCenters(data, labels)

        mergeClosest(clusters, data, labels)
