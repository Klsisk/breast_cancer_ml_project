select * from cancerdata;

COPY canderdata(diagnosis, radius_mean, texture_mean, perimeter_mean, area_mean, smoothness_mean, compactness_mean, concavity_mean, concave_points_mean, symmetry_mean, fractal_dimension_mean)
FROM 'C:\Users\klsis\Google Drive\Homework\breast_cancer_project\BCData\breast_cancer.csv'
DELIMITER ','
CSV HEADER;