/**
 * Dataset Upload Page
 */
import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { datasetAPI } from '../services/api';

export default function DatasetUpload() {
  const navigate = useNavigate();
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    dataset_type: 'retinal',
    contains_phi: false,
    anonymized: true,
  });
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [progress, setProgress] = useState(0);

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (!file) {
      alert('Please select a file');
      return;
    }

    setUploading(true);
    setProgress(10);

    try {
      // Create dataset
      const response = await datasetAPI.create(formData);
      const datasetId = response.data.id;
      setProgress(30);

      // Upload file
      await datasetAPI.upload(datasetId, file);
      setProgress(100);

      setTimeout(() => {
        navigate('/datasets');
      }, 500);
    } catch (error) {
      console.error('Upload error:', error);
      alert('Upload failed: ' + (error.response?.data?.detail || 'Unknown error'));
      setUploading(false);
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card">
        <h2 className="text-2xl font-bold text-gray-900 mb-6">Upload Dataset</h2>

        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dataset Name *
            </label>
            <input
              type="text"
              required
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="input"
              placeholder="My Retinal Dataset"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Description
            </label>
            <textarea
              value={formData.description}
              onChange={(e) => setFormData({ ...formData, description: e.target.value })}
              className="input"
              rows={3}
              placeholder="Description of your dataset..."
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">
              Dataset Type *
            </label>
            <select
              value={formData.dataset_type}
              onChange={(e) => setFormData({ ...formData, dataset_type: e.target.value })}
              className="input"
            >
              <option value="retinal">Retinal</option>
              <option value="ct">CT Scan</option>
              <option value="mri">MRI</option>
              <option value="xray">X-Ray</option>
              <option value="ultrasound">Ultrasound</option>
              <option value="pathology">Pathology</option>
              <option value="other">Other</option>
            </select>
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload File *
            </label>
            <div className="border-2 border-dashed border-gray-300 rounded-lg p-6 text-center hover:border-primary-500 transition-colors">
              <input
                type="file"
                onChange={(e) => setFile(e.target.files[0])}
                className="hidden"
                id="file-upload"
                accept=".tfrecord,.png,.jpg,.jpeg,.dcm"
              />
              <label htmlFor="file-upload" className="cursor-pointer">
                <svg className="mx-auto h-12 w-12 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <p className="mt-2 text-sm text-gray-600">
                  {file ? file.name : 'Click to upload or drag and drop'}
                </p>
                <p className="text-xs text-gray-500 mt-1">
                  TFRecord, PNG, JPG, or DICOM files
                </p>
              </label>
            </div>
          </div>

          <div className="space-y-2">
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData.anonymized}
                onChange={(e) => setFormData({ ...formData, anonymized: e.target.checked })}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="ml-2 text-sm text-gray-700">Data is anonymized</span>
            </label>
            <label className="flex items-center">
              <input
                type="checkbox"
                checked={formData.contains_phi}
                onChange={(e) => setFormData({ ...formData, contains_phi: e.target.checked })}
                className="rounded border-gray-300 text-primary-600 focus:ring-primary-500"
              />
              <span className="ml-2 text-sm text-gray-700">Contains PHI (Protected Health Information)</span>
            </label>
          </div>

          {uploading && (
            <div>
              <div className="w-full bg-gray-200 rounded-full h-2">
                <div
                  className="bg-primary-600 h-2 rounded-full transition-all duration-300"
                  style={{ width: `${progress}%` }}
                ></div>
              </div>
              <p className="text-sm text-gray-600 mt-2 text-center">Uploading... {progress}%</p>
            </div>
          )}

          <div className="flex space-x-4">
            <button
              type="submit"
              disabled={uploading}
              className="flex-1 btn btn-primary disabled:opacity-50"
            >
              {uploading ? 'Uploading...' : 'Upload Dataset'}
            </button>
            <button
              type="button"
              onClick={() => navigate('/datasets')}
              className="btn btn-secondary"
            >
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
