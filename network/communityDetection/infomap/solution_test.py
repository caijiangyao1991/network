import clustering
import os

def solution_test_main():
    filenames = ("solution_test1.txt", "solution_test2.txt", "solution_test3.txt")

    expected_partition1 = (('A', 'B', 'C'), ('D', 'E', 'F'))
    expected_partition2 = (('0', '1', '2', '3', '4', '5'), ('6', '7', '8', '9', '10', '11', '12'), ('13', '14', '15', '16'))
    expected_partition3 = (('0', '1', '2', '3'), ('4', '5', '6', '7'), ('8', '9', '10', '11'), ('12', '13', '14', '15'))
    expected_partitions = (expected_partition1, expected_partition2, expected_partition3)

    for filename, expected_partition in zip(filenames, expected_partitions):
        clusters_partition = clustering.clustering(filename, "solution_test_output.txt")
        for cluster in expected_partition:
            number = clusters_partition[cluster[0]]
            for node in cluster:
                if clusters_partition[node] != number:
                    os.remove("solution_test_output.txt")
                    return False
        labels_clusters = []
        for cluster in expected_partition:
            if clusters_partition[cluster[0]] in labels_clusters:
                os.remove("solution_test_output.txt")
                return False
            else:
                labels_clusters.append(clusters_partition[cluster[0]])

    os.remove("solution_test_output.txt")
    return True

if __name__ == "__main__":
    test_result = solution_test_main()
    if test_result:
        print("Solution test was successful")
    else:
        print("Solution test was not successful")
