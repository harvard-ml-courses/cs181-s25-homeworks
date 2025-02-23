import numpy as np

'''
Instructions for using this Autograder:

0. In this problem, this autograder is used as a sanity check for your classification implementations.
1. Make sure it is in the same immediate directory/folder as your implementation file.
2. Run this after you have implemented the SoftmaxRegression and KNNClassifier classes.
'''


def test_p3_softmax(vanilla_model, basis_model):
    vanilla_weights = np.array([[ 12.81489045,   3.56673263, -15.6785839 ],
       [  1.99837183,   0.84314957,   7.18119962],
       [-11.81326227,  -1.95369263,  10.95357383]])
    basis_weights = np.array([[ -9.2712308 ,   7.80486783,  -1.0546503 ],
       [ -5.57244059,   5.26416968,   0.48630338],
       [ 17.84367139, -10.61284795,   3.02453648]])

    student_v_weights = vanilla_model.W
    student_basis_weights = basis_model.W
    
    assert np.allclose(student_v_weights, vanilla_weights), f"Failed for softmax regression: incorrect weights"
    assert np.allclose(student_basis_weights, basis_weights), f"Failed for basis regression: incorrect weights"
    
    print("Passed softmax regression tests")


def test_p3_knn(knn_model_1, knn_model_5):
    points = np.asarray([(0, -1.5), (-6,-1)])
    
    knn_1_soln_preds = np.array([0, 0])
    knn_5_soln_preds = np.array([0, 2])
    
    knn_1_preds = knn_model_1.predict(points)
    knn_5_preds = knn_model_5.predict(points)
    
    knn_soln_distances = np.array([[ 5.06027778,  3.28387778,  1.93004444,  2.30444444,  3.10134444,
         3.88401111,  4.97084444,  6.47751111,  7.73      ,  9.57987778,
        13.10351111, 17.7025    , 25.26801111,  4.75694444,  5.57694444,
         6.63267778,  8.47254444,  9.62777778, 11.29361111,  6.11361111,
         6.52111111,  7.0625    ,  8.84      , 10.5536    , 11.12987778,
        13.01777778, 14.61137778],
       [ 0.42694444,  0.87721111,  3.37337778,  5.98777778,  8.38801111,
         9.83067778, 11.83417778, 14.51417778, 16.58      , 19.30654444,
        24.40684444, 31.0025    , 42.12134444,  8.32361111,  8.41027778,
         8.97934444,  9.54587778, 10.24444444, 11.12694444,  0.58027778,
         1.00444444,  1.5625    ,  2.89      ,  4.2436    ,  5.82321111,
         7.40111111,  8.75471111]])

    distances = np.array([[knn_model_1.distance(point, x) for x in knn_model_1.X] for point in points])
    
    assert np.allclose(knn_soln_distances, distances), f"Failed during distance calculation"
    assert np.allclose(knn_1_soln_preds, knn_1_preds), f"Failed for k = 1 predictions"
    assert np.allclose(knn_5_soln_preds, knn_5_preds), f"Failed for k = 5 predictions"
    
    print("Passed KNN tests")
