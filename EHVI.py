"""
Expected HyperVolume Improvement: for any p-dimension, using `numpy`
Use package: `threading` to do multithreading in calculation

Version: 2022-05-13
Author: Quan-En, Li

- class: EHVI
    -- calculate_iEI(self, distribution_info: list): -> float
        where
        distribution_info=[
        mean_vector: np.array (m, ),
        var_vector: np.array (m, ),
        ]
        m = number of objective
    
    -- calculate_rEI(self, distribution_info: list): -> float
        where
        distribution_info=[
        mean_vector: np.array (m, ),
        cov_matrix: np.array (m, m, ),
        ]
        m = number of objective
    
    -- decide_active_region_f(self, pareto_set: list): -> corners_list: list
        where
        corners_list = [
        active_lower_corner_list,
        active_upper_corner_list,
        ]
    
    -- get_ref_points(self, corners_list: list): -> list
    -- get_all_volume_L(self, corners_list: list): -> list

Process:
    Declare `Criterion` class:
        (1) Setting: pareto set, reference point(max boundary).
        (2) Treat pareto set as input to `decide_active_region_f` get all subsection lower corner value and upper corner value.
        (3) Use corners list and `get_ref_points` to get reference point of each subsection.
        (4) Use corners list and `get_all_volume_L` to get volume-L if each subsection.
    Calculation:
        (1) Use `calculate_iEI` to calculate EI if objective functions are independent.
        (2) Use `calculate_rEI` to calculate EI if objective functions are related.
"""


# Check how many core and use 1/3
import os
TOTAL_CPU_NUM = os.cpu_count()
USE_CPU_NUM = int(TOTAL_CPU_NUM / 3 if TOTAL_CPU_NUM / 3 > 1 else 1)
import threading

import numpy as np
from numpy.linalg import det

from scipy.stats import multivariate_normal as mvn

from pygmo import hypervolume # python version==3.7.4
from itertools import product


# Monte-Carlo Integration
def Monte_Carlo_Int(int_fn, int_range:list, size:int=50000, *args):

    int_range_array = np.array(int_range)
    int_dim = int_range_array.shape[0]

    samples = np.zeros((size, int_dim))

    for d in range(int_dim):
        scale = int_range_array[d,1] - int_range_array[d,0]
        offset = int_range_array[d,0]
        samples[:,d] = np.random.rand(size) * scale + offset

    function_values = int_fn(samples, *args)

    scales = int_range_array[:, 1] - int_range_array[:, 0]
    volume = np.prod(scales)
    
    int_result = volume * np.sum(function_values) / size

    return int_result

# integral function
def int_fn(x, ref_point, upper_corner, volume_L, mean_v, cov_mat):

    part1 = np.prod(ref_point - x, axis=1) - np.prod(ref_point - upper_corner) + volume_L
    prat2 = mvn.pdf(x, mean=mean_v, cov=cov_mat)

    return part1 * prat2

def Main_Int(result, index, *args):
    result[index] = Monte_Carlo_Int(*args)

class Criterion(object):
    def __init__(self, pareto_set: list, boundary_point: np.array):
        self.pareto_set = pareto_set
        self.boundary_point = boundary_point
        (
            self.active_lower_corners_list,
            self.active_upper_corners_list,
        ) = self.decide_active_region_f(pareto_set)

        self.ref_points_list = self.get_ref_points(
            [self.active_lower_corners_list, self.active_upper_corners_list]
        )
        self.all_volume_L_list = self.get_all_volume_L(
            [self.active_lower_corners_list, self.active_upper_corners_list]
        )

        self.num_of_area = len(self.ref_points_list)

    def forward(self, distribution_info: list, is_independent=True):
        if is_independent: return self.calculate_iEI(distribution_info)
        else: return self.calculate_rEI(distribution_info)

    def calculate_iEI(self, distribution_info: list):
        #### Assume objective are independent: easy way to solve ####

        # mean vector(m), variance vector(m) = distribution information
        mean_vector, var_vector = distribution_info
        std_vector = np.sqrt(var_vector)

        # Avoid standard deviation equal to zero
        zero_condition = std_vector <= 1e-3
        all_zero_condition = zero_condition.all()
        std_vector[zero_condition] = 1e-3

        if all_zero_condition.item():
            return np.array([0])

        # Calculate expected improvement of each area
        delta_indep_list = []

        for i in range(self.num_of_area):

            # Get the ref_point: array (m, )
            ref_point = self.ref_points_list[i]

            # Get the lower corner value & upper corner value
            # lower_corner_value: array (m, )
            # upper_corner_value: array (m, )
            lower_corner_value = self.active_lower_corners_list[i]
            upper_corner_value = self.active_upper_corners_list[i]

            # Get volume-L: pure float
            volume_L_value = self.all_volume_L_list[i]

            # Calculate EI of each subsection

            # Standardize corner value
            standardize_lower_corner = (lower_corner_value - mean_vector) / std_vector
            standardize_upper_corner = (upper_corner_value - mean_vector) / std_vector

            # 'Cumulative density' & 'Probability density' at lower corner
            lower_corner_cdf = mvn.cdf(standardize_lower_corner)
            lower_corner_pdf = mvn.pdf(standardize_lower_corner)

            # 'Cumulative density' & 'Probability density' at upper corner
            upper_corner_cdf = mvn.cdf(standardize_upper_corner)
            upper_corner_pdf = mvn.pdf(standardize_upper_corner)

            # Cumulative density between [lower corner, upper corner]
            interval_probability_each_dim = upper_corner_cdf - lower_corner_cdf
            interval_probability_each_dim = interval_probability_each_dim

            # Product of each dimension cumulative density
            interval_probability = np.prod(interval_probability_each_dim)

            # Calculate EI
            Q3 = volume_L_value * interval_probability

            Q2 = np.prod(ref_point - upper_corner_value) * interval_probability

            Q1_left_sides = (
                ref_point - mean_vector
            ) * upper_corner_cdf + std_vector * upper_corner_pdf
            Q1_right_sides = (
                ref_point - mean_vector
            ) * lower_corner_cdf + std_vector * lower_corner_pdf
            Q1 = np.prod(Q1_left_sides - Q1_right_sides)

            delta_indep = Q1 - Q2 + Q3
            delta_indep_list.append(delta_indep)

        return np.array(delta_indep_list).sum()

    def calculate_rEI(self, distribution_info: list):
        #### Assume objective are related ####

        # mean vector(m), covariance matrix(m, m) = distribution information
        mean_vector, covar_matrix = distribution_info

        # avoid standard deviation equal to zero
        zero_condition = np.diag(covar_matrix) <= 1e-3
        all_zero_condition = zero_condition.all()
        
        if all_zero_condition.item():
            return np.array([0])
        
        if zero_condition.any().item():
            covar_matrix[zero_condition, :] = 0
            covar_matrix[:, zero_condition] = 0
            covar_matrix[np.diag(zero_condition)] = 1e-3
            

        # avoid covar_matrix is singular
        covar_matrix_det = det(covar_matrix).item()
        if covar_matrix_det < 1e-10:
            covar_matrix = covar_matrix + 1e-6 * np.ones(covar_matrix.shape)
            # covar_matrix = np.diag(np.diag(covar_matrix)) # transform to independent

        std_vector = np.sqrt(np.diag(covar_matrix))
        


        # Get minimum integrate boundary (avoid -inf boundary)
        minimum_boundary = np.min(np.row_stack(self.active_upper_corners_list), axis=0) - 4 * std_vector

        # Compute the integral boundary
        int_boundary_list = []
        for lower_corner_value, upper_corner_value in zip(self.active_lower_corners_list, self.active_upper_corners_list):
            boundary_list = [np.copy(lower_corner_value), np.copy(upper_corner_value)]
            # boundary_list = [lower_corner_value, upper_corner_value]

            # -inf condition: replace to minimum boundary
            neg_inf_cond = (lower_corner_value == -float("inf"))
            boundary_list[0][neg_inf_cond] = minimum_boundary[neg_inf_cond]

            int_boundary = np.column_stack(boundary_list).tolist()
            int_boundary_list.append(int_boundary)
        
        # Calculate expected improvement of each area
        delta_relate_list = [None] * self.num_of_area
        
        thread_pools = [None] * USE_CPU_NUM

        for i in range(self.num_of_area):
            # Do multithreading
            
            ## Define in pools
            thread_pools[i % USE_CPU_NUM] = threading.Thread(
                target=Main_Int,
                args=(
                    ## args for `Main_Int`
                    delta_relate_list, i,
                    ## args for `Monte_Carlo_Int`
                    int_fn, int_boundary_list[i], 50000,
                    ## args for `int_fn`
                    self.ref_points_list[i],
                    self.active_upper_corners_list[i],
                    self.all_volume_L_list[i],
                    mean_vector, covar_matrix,
                )
            )
            thread_pools[i % USE_CPU_NUM].start()
            
            ## Join
            if (i + 1) % USE_CPU_NUM == 0:
                for i in range(USE_CPU_NUM):thread_pools[i].join()
                thread_pools = [None] * USE_CPU_NUM
            elif (i + 1) == self.num_of_area:
                for i in range(i % USE_CPU_NUM + 1):thread_pools[i].join()
        
        # return delta_relate_list
        return np.array(delta_relate_list).sum()

    def ei_contribute(self):
        hv = hypervolume(list(map(lambda x: x.tolist(), self.pareto_set)))
        ei_contribute_value = hv.compute(self.boundary_point.tolist())
        return ei_contribute_value

    def decide_active_region_f(self, pareto_set: list):

        # n = |pareto_set|
        n = len(pareto_set)

        # m = dimensions of objective function
        m = pareto_set[0].shape[0]

        # negative infinity array
        neg_inf_array = np.array([-float("inf")] * m)

        # get partition value of each dimension
        partition_matrix = np.row_stack(pareto_set + [neg_inf_array] + [self.boundary_point])

        each_dim_value_list = np.split(
            partition_matrix, indices_or_sections=partition_matrix.shape[1], axis=1
        )
        each_dim_unique_value_list = list(
            map(lambda x: np.unique(x.reshape(-1)).tolist(), each_dim_value_list)
        )

        # get list of `all` lower_corner_value and upper_corner_value
        all_lower_corners_list = list(
            product(*list(map(lambda x: x[:-1], each_dim_unique_value_list)))
        )

        all_upper_corners_list = list(
            product(*list(map(lambda x: x[1:], each_dim_unique_value_list)))
        )

        all_lower_corners_list = list(map(lambda x: np.array(x), all_lower_corners_list))
        all_upper_corners_list = list(map(lambda x: np.array(x), all_upper_corners_list))

        # get list of `active` lower_corner_value and upper_corner_value
        active_lower_corners_list = []
        active_upper_corners_list = []

        inactive_lower_corners_list = []
        inactive_upper_corners_list = []

        pareto_matrix = np.row_stack(pareto_set)
        for lower_corner_value, upper_corner_value in zip(
            all_lower_corners_list, all_upper_corners_list
        ):
            cond_1 = (lower_corner_value <= pareto_matrix).all(axis=1)
            cond_2 = (lower_corner_value < pareto_matrix).any(axis=1)
            if any(cond_1 * cond_2):
                active_lower_corners_list.append(lower_corner_value)
                active_upper_corners_list.append(upper_corner_value)
            else:
                inactive_lower_corners_list.append(lower_corner_value)
                inactive_upper_corners_list.append(upper_corner_value)
        
        return active_lower_corners_list, active_upper_corners_list

    def get_ref_points(self, corners_list: list):

        active_lower_corners_list, active_upper_corners_list = corners_list
        active_upper_corners_matrix = np.row_stack(active_upper_corners_list)

        ref_points_list = []

        for upper_corner_value in active_upper_corners_list:
            cond = (upper_corner_value <= active_upper_corners_matrix).all(axis=1)
            ref_point_value = np.max(active_upper_corners_matrix[cond, :], axis=0)
            ref_points_list.append(ref_point_value)

        return ref_points_list

    def get_all_volume_L(self, corners_list: list):
        active_lower_corners_list, active_upper_corners_list = corners_list

        # empty list to store volume-L
        volume_L_value_list = []

        # row-stack to matrix
        active_lower_corners_matrix = np.row_stack(active_lower_corners_list)
        active_upper_corners_matrix = np.row_stack(active_upper_corners_list)

        # for lower_corner_value, upper_corner_value in zip(
        #     active_lower_corners_list, active_upper_corners_list
        # ):
        for upper_corner_value in active_upper_corners_list:
            cond_1 = np.all(upper_corner_value <= active_lower_corners_matrix, axis=1)
            cond_2 = np.all(upper_corner_value <= active_upper_corners_matrix, axis=1)
            cond = cond_1 * cond_2

            if any(cond):
                sub_volume_L = np.prod(
                    active_upper_corners_matrix[cond, :]
                    - active_lower_corners_matrix[cond, :],
                    axis=1,
                ).sum()
                volume_L_value_list.append(sub_volume_L)
            else:
                volume_L_value_list.append(0)
        return volume_L_value_list