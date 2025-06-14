use kiddo::{ImmutableKdTree, SquaredEuclidean};
use numpy::{PyArray2, PyReadonlyArray2};
use pyo3::prelude::*;
use rayon::prelude::*;

/// A Python wrapper for kiddo's ImmutableKdTree
#[pyclass]
pub struct PyKdTree {
    dimensions: usize,

    points_2d: Option<Vec<[f32; 2]>>,
    tree_2d: Option<ImmutableKdTree<f32, 2>>,

    points_3d: Option<Vec<[f32; 3]>>,
    tree_3d: Option<ImmutableKdTree<f32, 3>>,
}

#[pymethods]
impl PyKdTree {
    /// Create a new k-d tree with the specified number of dimensions and points
    ///
    /// Args:
    ///     dimensions: The number of dimensions (2 or 3)
    ///     points: A 2D numpy array where each row is a point
    #[new]
    pub fn new(dimensions: usize, points: PyReadonlyArray2<f32>) -> PyResult<Self> {
        if !(2..=3).contains(&dimensions) {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Dimensions must be 2 or 3",
            ));
        }

        let points_array = points.as_array();
        if points_array.shape()[1] != dimensions {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Points must have {} dimensions",
                dimensions
            )));
        }

        let mut tree = PyKdTree {
            dimensions,
            points_2d: None,
            tree_2d: None,
            points_3d: None,
            tree_3d: None,
        };

        match dimensions {
            2 => {
                let pts: Vec<[f32; 2]> = points_array
                    .outer_iter()
                    .map(|row| [row[0], row[1]])
                    .collect();
                tree.tree_2d = if !pts.is_empty() {
                    Some(ImmutableKdTree::new_from_slice(&pts))
                } else {
                    None
                };
                tree.points_2d = Some(pts);
            }
            3 => {
                let pts: Vec<[f32; 3]> = points_array
                    .outer_iter()
                    .map(|row| [row[0], row[1], row[2]])
                    .collect();
                tree.tree_3d = if !pts.is_empty() {
                    Some(ImmutableKdTree::new_from_slice(&pts))
                } else {
                    None
                };
                tree.points_3d = Some(pts);
            }
            _ => unreachable!(),
        }

        Ok(tree)
    }

    /// Find all points within a specified distance of multiple query points
    ///
    /// Args:
    ///     distance: The maximum distance to search within
    ///     query_points: A 2D numpy array where each row is a query point
    ///     parallel: Whether to use parallel processing with rayon (default: false)
    ///
    /// Returns:
    ///     A 2D numpy array where each row is [query_index, point_index, distance]
    #[pyo3(signature = (distance, query_points, parallel = false))]
    pub fn within_unsorted(
        &self,
        py: Python,
        distance: f32,
        query_points: PyReadonlyArray2<f32>,
        parallel: bool,
    ) -> PyResult<PyObject> {
        let queries_array = query_points.as_array();
        if queries_array.shape()[1] != self.dimensions {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "Query points must have {} dimensions",
                self.dimensions
            )));
        }

        let squared_distance = distance * distance;
        let num_queries = queries_array.shape()[0];

        macro_rules! process_queries {
            ($tree:expr, $query_array_expr:expr) => {{
                let tree = $tree.as_ref().ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Tree not initialized")
                })?;

                let process_query = |query_idx: usize| -> Vec<(usize, usize, f32)> {
                    let query_array =
                        $query_array_expr(queries_array.row(query_idx).as_slice().unwrap());
                    tree.within_unsorted::<SquaredEuclidean>(&query_array, squared_distance)
                        .into_iter()
                        .map(|r| (query_idx, r.item as usize, r.distance.sqrt()))
                        .collect()
                };

                if parallel {
                    let chunk_size = (num_queries / rayon::current_num_threads()).max(1);
                    (0..num_queries)
                        .into_par_iter()
                        .chunks(chunk_size)
                        .map(|chunk| {
                            let mut local_results = Vec::new();
                            for query_idx in chunk {
                                local_results.extend(process_query(query_idx));
                            }
                            local_results
                        })
                        .flatten()
                        .collect()
                } else {
                    let mut all_results = Vec::new();
                    for query_idx in 0..num_queries {
                        all_results.extend(process_query(query_idx));
                    }
                    all_results
                }
            }};
        }

        let all_results: Vec<(usize, usize, f32)> = match self.dimensions {
            2 => process_queries!(self.tree_2d, |slice: &[f32]| [slice[0], slice[1]]),
            3 => process_queries!(self.tree_3d, |slice: &[f32]| [slice[0], slice[1], slice[2]]),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported dimensions",
                ))
            }
        };

        let array_data: Vec<Vec<f32>> = all_results
            .into_iter()
            .map(|(qi, pi, dist)| vec![qi as f32, pi as f32, dist])
            .collect();

        Ok(PyArray2::from_vec2(py, &array_data)?.into())
    }

    /// Find all pairs of points within a specified distance
    ///
    /// Args:
    ///     distance: The maximum distance between pairs
    ///     parallel: Whether to use parallel processing with rayon (default: false)
    ///
    /// Returns:
    ///     A 2D numpy array where each row is [point_index_i, point_index_j, distance] where i < j
    #[pyo3(signature = (distance, parallel = false))]
    pub fn query_pairs(&self, py: Python, distance: f32, parallel: bool) -> PyResult<PyObject> {
        let squared_distance = distance * distance;

        macro_rules! process_dimension {
            ($tree:expr, $points:expr) => {{
                let (tree, points) = ($tree.as_ref(), $points.as_ref());
                let tree = tree.ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Tree not initialized")
                })?;
                let points = points.ok_or_else(|| {
                    pyo3::exceptions::PyRuntimeError::new_err("Points not initialized")
                })?;
                let n_points = points.len();

                let process_point = |i: usize| -> Vec<(u64, u64, f32)> {
                    tree.within_unsorted::<SquaredEuclidean>(&points[i], squared_distance)
                        .into_iter()
                        .filter_map(|result| {
                            let j = result.item;
                            (j > i as u64).then(|| (i as u64, j, result.distance.sqrt()))
                        })
                        .collect()
                };

                if parallel {
                    let chunk_size = (n_points / rayon::current_num_threads()).max(1);
                    (0..n_points)
                        .into_par_iter()
                        .chunks(chunk_size)
                        .map(|chunk| {
                            let mut local_pairs = Vec::new();
                            for i in chunk {
                                local_pairs.extend(process_point(i));
                            }
                            local_pairs
                        })
                        .flatten()
                        .collect()
                } else {
                    let mut all_pairs = Vec::new();
                    for i in 0..n_points {
                        all_pairs.extend(process_point(i));
                    }
                    all_pairs
                }
            }};
        }

        let all_pairs: Vec<(u64, u64, f32)> = match self.dimensions {
            2 => process_dimension!(self.tree_2d, self.points_2d),
            3 => process_dimension!(self.tree_3d, self.points_3d),
            _ => {
                return Err(pyo3::exceptions::PyValueError::new_err(
                    "Unsupported dimensions",
                ))
            }
        };

        if all_pairs.is_empty() {
            return Ok(PyArray2::<f32>::zeros(py, [0, 3], false).into());
        }

        let array_data: Vec<Vec<f32>> = all_pairs
            .into_iter()
            .map(|(i, j, dist)| vec![i as f32, j as f32, dist])
            .collect();

        Ok(PyArray2::from_vec2(py, &array_data)?.into())
    }
    /// Get the number of points in the tree
    pub fn size(&self) -> usize {
        match self.dimensions {
            2 => self.tree_2d.as_ref().map_or(0, |t| t.size() as usize),
            3 => self.tree_3d.as_ref().map_or(0, |t| t.size() as usize),
            _ => 0,
        }
    }

    /// Get the number of dimensions
    pub fn dimensions(&self) -> usize {
        self.dimensions
    }
}

/// A Python module implemented in Rust.
#[pymodule]
fn kiddo_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyKdTree>()?;
    Ok(())
}
