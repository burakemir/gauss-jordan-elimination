// Copyright 2022 Burak Emir

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

use std::vec::Vec;

pub enum GaussEliminationOption {
    JustEchelon,
    PrepareReduce
}

// matrix is represented as vector of rows, a row being a vector of columns.
pub fn gauss_elimination(matrix: &mut Vec<Vec<f32>>, option: GaussEliminationOption) {
    fn find_pivot(matrix: &mut Vec<Vec<f32>>, d: usize) -> Option<usize> {
        return (d..matrix.len()).find(|&i| matrix[i][d] != 0f32);
    }
    let nrows = matrix.len();
    for c in 0..nrows {
        match find_pivot(matrix, c) {
            None => {}
            Some(i) => {
                for row in i + 1..nrows {
                    let factor = matrix[row][c] / matrix[i][c];
                    for col in c..matrix[row].len() {
                        matrix[row][col] -= factor * matrix[i][col]
                    }
                }
                // Move pivot to row c, in order to get a "real" echelon form.
                if c != i {
                    matrix.swap(i, c);
                }
                if matches!(option, GaussEliminationOption::PrepareReduce) {
                    // normalize the pivot to 1.0
                    let factor = 1.0 / matrix[c][c];
                    for col in c..matrix[c].len() {
                        matrix[c][col] *= factor
                    }
                }
            }
        }
    }
}

pub fn gauss_jordan_elimination(matrix: &mut Vec<Vec<f32>>) {
    gauss_elimination(matrix, GaussEliminationOption::PrepareReduce);
    let nrows = matrix.len();

    fn find_pivot(matrix: &mut Vec<Vec<f32>>, d: usize) -> Option<usize> {
        return (d..matrix.len()).find(|&i| matrix[d][i] != 0f32);
    }

    for d in (1..nrows).rev() {
        match find_pivot(matrix, d) {
            None => {}
            Some(i) => {
                for row in (0..d).rev() {
                    let factor = matrix[row][i];
                    for col in d..matrix[i].len() {
                        matrix[row][col] -= factor * matrix[d][col]
                    }
                }
            }
        }
    }
}


#[cfg(feature = "generic_calculation")] 
pub fn gauss_jordan_elimination_generic<T>(matrix: &mut Vec<Vec<T>>)
where T: num::traits::One + num::traits::Zero + std::ops::AddAssign + std::ops::SubAssign 
+ std::ops::Mul<Output = T> + std::ops::MulAssign + std::ops::Div<Output = T> + Clone + Copy 
+ for<'a> std::ops::Add<&'a T> + for<'a> std::ops::Mul<&'a T, Output =  T>,
for<'a> &'a T: std::ops::Mul<T, Output = T>{

gauss_elimination_generic(matrix, GaussEliminationOption::PrepareReduce);
let nrows = matrix.len();

fn find_pivot <T>(matrix: &mut Vec<Vec<T>>, d: usize) -> Option<usize>
where T: num::traits::Zero + std::ops::AddAssign + std::ops::SubAssign {
    return (d..matrix.len()).find(|&i| matrix[d][i].is_zero() == false);
}

for d in (1..nrows).rev() {
    match find_pivot::<T>(matrix, d) {
        None => {}
        Some(i) => {
            for row in (0..d).rev() {
                let factor = matrix[row][i];
                for col in d..matrix[i].len() {
                    let number = matrix[d][col];
                    matrix[row][col] -= factor * number
                }
            }
        }
    }
}
}



//

#[cfg(feature = "generic_calculation")] 
pub fn gauss_elimination_generic<T>(matrix: &mut Vec<Vec<T>>, option: GaussEliminationOption) 
where T: num::traits::One + num::traits::Zero + std::ops::AddAssign + std::ops::SubAssign 
+ std::ops::Mul<Output = T> + std::ops::MulAssign + std::ops::Div<Output = T> + Clone + Copy{
    fn find_pivot <T: num::traits::Zero >(matrix: &mut Vec<Vec<T>>, d: usize) -> Option<usize> {
        return (d..matrix.len()).find(|&i| matrix[i][d].is_zero() == false);
    }
    let nrows = matrix.len();
    for c in 0..nrows {
        match find_pivot::<T>(matrix, c) {
            None => {}
            Some(i) => {
                for row in i + 1..nrows {
                    let factor = matrix[row][c] / matrix[i][c];
                    for col in c..matrix[row].len() {
                        let number = matrix[i][col];
                        matrix[row][col] -= factor * number
                    }
                }
                // Move pivot to row c, in order to get a "real" echelon form.
                if c != i {
                    matrix.swap(i, c);
                }
                if matches!(option, GaussEliminationOption::PrepareReduce) {
                    // normalize the pivot to 1.0
                    let factor = T::one() / matrix[c][c];
                    for col in c..matrix[c].len() {
                        matrix[c][col] *= factor
                    }
                }
            }
        }
    }
}
    


#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn test_gauss_elimination_echelon() {
        let mut matrix = vec![
            vec![1.0, 2.0, 1.0, 10.0],
            vec![2.0, 3.0, 2.0, 12.0],
            vec![3.0, 1.0, 4.0, 11.0],
        ];
        gauss_elimination(&mut matrix, GaussEliminationOption::JustEchelon);
        assert_eq!(
            matrix,
            vec![
                vec![1.0, 2.0, 1.0, 10.0],
                vec![0.0, -1.0, 0.0, -8.0],
                vec![0.0, 0.0, 1.0, 21.0],
            ]
        );
    }

    #[test]
    fn test_gauss_elimination_reduce() {
        let mut matrix = vec![
            vec![1.0, 2.0, 1.0, 10.0],
            vec![2.0, 3.0, 2.0, 12.0],
            vec![3.0, 1.0, 4.0, 11.0],
        ];
        gauss_elimination(&mut matrix, GaussEliminationOption::PrepareReduce);
        assert_eq!(
            matrix,
            vec![
                vec![1.0, 2.0, 1.0, 10.0],
                vec![0.0, 1.0, 0.0, 8.0],
                vec![0.0, 0.0, 1.0, 21.0],
            ]
        );
    }

    #[test]
    fn test_gauss_jordan_elimination() {
        let mut matrix = vec![
            vec![1.0, 2.0, 1.0, 10.0],
            vec![2.0, 3.0, 2.0, 12.0],
            vec![3.0, 1.0, 4.0, 11.0],
        ];
        gauss_jordan_elimination(&mut matrix);
        assert_eq!(
            matrix,
            vec![
                vec![1.0, 0.0, 0.0, -27.0],
                vec![0.0, 1.0, 0.0, 8.0],
                vec![0.0, 0.0, 1.0, 21.0],
            ]
        );
    }

    #[test]
    fn test_gauss_elimination_zero() {
        let mut matrix = vec![
            vec![1.0, 2.0, 1.0, 10.0],
            vec![0.0, 0.0, 2.0, 12.0],
            vec![3.0, 1.0, 4.0, 11.0],
        ];
        gauss_elimination(&mut matrix, GaussEliminationOption::JustEchelon);
        assert_eq!(
            matrix,
            vec![
                vec![1.0, 2.0, 1.0, 10.0],
                vec![0.0, -5.0, 1.0, -19.0],
                vec![0.0, 0.0, 2.0, 12.0],
            ]
        );
    }


    #[cfg(feature = "generic_calculation")] 
    #[test]
    fn test_gauss_elimination_echelon_generic() {
        let mut matrix = vec![
            vec![1.0, 2.0, 1.0, 10.0],
            vec![2.0, 3.0, 2.0, 12.0],
            vec![3.0, 1.0, 4.0, 11.0],
        ];
        gauss_elimination_generic::<f64>(&mut matrix, GaussEliminationOption::JustEchelon);
        assert_eq!(
            matrix,
            vec![
                vec![1.0, 2.0, 1.0, 10.0],
                vec![0.0, -1.0, 0.0, -8.0],
                vec![0.0, 0.0, 1.0, 21.0],
            ]
        );
    }

    #[cfg(feature = "generic_calculation")] 
    #[test]
    fn test_gauss_elimination_reduce_generic() {
        let mut matrix = vec![
            vec![1.0, 2.0, 1.0, 10.0],
            vec![2.0, 3.0, 2.0, 12.0],
            vec![3.0, 1.0, 4.0, 11.0],
        ];
        gauss_elimination_generic::<f64>(&mut matrix, GaussEliminationOption::PrepareReduce);
        assert_eq!(
            matrix,
            vec![
                vec![1.0, 2.0, 1.0, 10.0],
                vec![0.0, 1.0, 0.0, 8.0],
                vec![0.0, 0.0, 1.0, 21.0],
            ]
        );
    }

    #[cfg(feature = "generic_calculation")] 
    #[test]
    fn test_gauss_jordan_elimination_generic() {
        let mut matrix = vec![
            vec![1.0, 2.0, 1.0, 10.0],
            vec![2.0, 3.0, 2.0, 12.0],
            vec![3.0, 1.0, 4.0, 11.0],
        ];
        gauss_jordan_elimination_generic::<f64>(&mut matrix);
        assert_eq!(
            matrix,
            vec![
                vec![1.0, 0.0, 0.0, -27.0],
                vec![0.0, 1.0, 0.0, 8.0],
                vec![0.0, 0.0, 1.0, 21.0],
            ]
        );
    }

    #[cfg(feature = "generic_calculation")] 
    #[test]
    fn test_gauss_elimination_zero_generic() {
        let mut matrix = vec![
            vec![1.0, 2.0, 1.0, 10.0],
            vec![0.0, 0.0, 2.0, 12.0],
            vec![3.0, 1.0, 4.0, 11.0],
        ];
        gauss_elimination_generic::<f64>(&mut matrix, GaussEliminationOption::JustEchelon);
        assert_eq!(
            matrix,
            vec![
                vec![1.0, 2.0, 1.0, 10.0],
                vec![0.0, -5.0, 1.0, -19.0],
                vec![0.0, 0.0, 2.0, 12.0],
            ]
        );
    }
}
