/**
 * MedSynth Platform - Datasets Page
 */
import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import { datasetAPI } from '../services/api';

export default function Datasets() {
  const [datasets, setDatasets] = useState([]);
  const [loading, setLoading] = useState(true);
  const [page, setPage] = useState(1);
  const [total, setTotal] = useState(0);

  useEffect(() => {
    loadDatasets();
  }, [page]);

  const loadDatasets = async () => {
    try {
      const response = await datasetAPI.list({ page, page_size: 10 });
      setDatasets(response.data.datasets);
      setTotal(response.data.total);
    } catch (error) {
      console.error('Error loading datasets:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleDelete = async (id) => {
    if (!window.confirm('Are you sure you want to delete this dataset?')) {
      return;
    }

    try {
      await datasetAPI.delete(id);
      loadDatasets();
    } catch (error) {
      console.error('Error deleting dataset:', error);
      alert('Failed to delete dataset');
    }
  };

  const getStatusColor = (status) => {
    const colors = {
      ready: 'badge-success',
      processing: 'badge-warning',
      uploading: 'badge-info',
      error: 'badge-danger',
      archived: 'badge-secondary',
    };
    return colors[status] || 'badge-secondary';
  };

  const formatFileSize = (bytes) => {
    if (!bytes) return 'N/A';
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(1024));
    return `${(bytes / Math.pow(1024, i)).toFixed(2)} ${sizes[i]}`;
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Datasets</h2>
          <p className="text-gray-600 mt-1">Manage your medical imaging datasets</p>
        </div>
        <Link to="/datasets/upload" className="btn btn-primary">
          <svg className="w-5 h-5 mr-2 inline" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          Upload Dataset
        </Link>
      </div>

      {/* Datasets Table */}
      {datasets.length === 0 ? (
        <div className="card text-center py-12">
          <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20 13V6a2 2 0 00-2-2H6a2 2 0 00-2 2v7m16 0v5a2 2 0 01-2 2H6a2 2 0 01-2-2v-5m16 0h-2.586a1 1 0 00-.707.293l-2.414 2.414a1 1 0 01-.707.293h-3.172a1 1 0 01-.707-.293l-2.414-2.414A1 1 0 006.586 13H4" />
          </svg>
          <h3 className="mt-2 text-lg font-medium text-gray-900">No datasets</h3>
          <p className="mt-1 text-gray-500">Get started by uploading your first dataset.</p>
          <Link to="/datasets/upload" className="btn btn-primary mt-4 inline-block">
            Upload Dataset
          </Link>
        </div>
      ) : (
        <div className="card overflow-hidden p-0">
          <table className="min-w-full divide-y divide-gray-200">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Name
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Type
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Status
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Size
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Images
                </th>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Created
                </th>
                <th className="px-6 py-3 text-right text-xs font-medium text-gray-500 uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-white divide-y divide-gray-200">
              {datasets.map((dataset) => (
                <tr key={dataset.id} className="hover:bg-gray-50">
                  <td className="px-6 py-4">
                    <div>
                      <div className="font-medium text-gray-900">{dataset.name}</div>
                      {dataset.description && (
                        <div className="text-sm text-gray-500">{dataset.description}</div>
                      )}
                    </div>
                  </td>
                  <td className="px-6 py-4">
                    <span className="text-sm text-gray-900 capitalize">{dataset.dataset_type}</span>
                  </td>
                  <td className="px-6 py-4">
                    <span className={`badge ${getStatusColor(dataset.status)}`}>
                      {dataset.status}
                    </span>
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900">
                    {formatFileSize(dataset.file_size)}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-900">
                    {dataset.num_images || 'N/A'}
                  </td>
                  <td className="px-6 py-4 text-sm text-gray-500">
                    {new Date(dataset.created_at).toLocaleDateString()}
                  </td>
                  <td className="px-6 py-4 text-right text-sm font-medium space-x-2">
                    <Link
                      to={`/training/create?dataset=${dataset.id}`}
                      className="text-primary-600 hover:text-primary-900"
                    >
                      Train
                    </Link>
                    <button
                      onClick={() => handleDelete(dataset.id)}
                      className="text-red-600 hover:text-red-900"
                    >
                      Delete
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Pagination */}
      {total > 10 && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-700">
            Showing {(page - 1) * 10 + 1} to {Math.min(page * 10, total)} of {total} results
          </p>
          <div className="flex space-x-2">
            <button
              onClick={() => setPage(page - 1)}
              disabled={page === 1}
              className="btn btn-secondary disabled:opacity-50"
            >
              Previous
            </button>
            <button
              onClick={() => setPage(page + 1)}
              disabled={page * 10 >= total}
              className="btn btn-secondary disabled:opacity-50"
            >
              Next
            </button>
          </div>
        </div>
      )}
    </div>
  );
}
