# Intelligibility-Detection

#Let's keep a tract of the experiments here

1) threshold_gaussian.py : - Sets the threshold for overlap of gaussians
2) threshold_multiprocess.py : - Calculates the threshold using multiprocessing.

To run: python threshold_multiprocess.py <number_of_cores> <version_of_match_score>. If the parameters are not mentioned, then number of cores = os.cpu_count() and version=1. 








Flow of exec:
1) threshold_gaussian.py 

Link for intersection of gaussians: https://stats.stackexchange.com/questions/311592/how-to-find-the-point-where-two-normal-distributions-intersect

2) plot_gaussians.py : plots the gaussians that have been generated in  threshold_gaussian.py
