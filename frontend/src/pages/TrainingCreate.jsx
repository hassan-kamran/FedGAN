import React, { useState, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { trainingAPI, datasetAPI } from '../services/api';

export default function TrainingCreate() {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [datasets, setDatasets] = useState([]);
  const [formData, setFormData] = useState({
    name: '',
    description: '',
    dataset_id: searchParams.get('dataset') || '',
    num_clients: 5,
    num_rounds: 100,
    batch_size: 32,
    learning_rate: 0.0002,
    latent_dim: 200,
  });

  useEffect(() => {
    loadDatasets();
  }, []);

  const loadDatasets = async () => {
    try {
      const response = await datasetAPI.list({ page: 1, page_size: 100 });
      setDatasets(response.data.datasets.filter(d => d.status === 'ready'));
    } catch (error) {
      console.error('Error loading datasets:', error);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    try {
      const response = await trainingAPI.create(formData);
      navigate(`/training/${response.data.id}`);
    } catch (error) {
      alert('Failed to create training job: ' + (error.response?.data?.detail || 'Unknown error'));
    }
  };

  return (
    <div className="max-w-2xl mx-auto">
      <div className="card">
        <h2 className="text-2xl font-bold mb-6">Create Training Job</h2>
        <form onSubmit={handleSubmit} className="space-y-6">
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Job Name *</label>
            <input
              type="text"
              required
              value={formData.name}
              onChange={(e) => setFormData({ ...formData, name: e.target.value })}
              className="input"
              placeholder="My Training Job"
            />
          </div>

          <div>
            <label className="block text-sm font-medium text-gray-700 mb-1">Dataset *</label>
            <select
              required
              value={formData.dataset_id}
              onChange={(e) => setFormData({ ...formData, dataset_id: e.target.value })}
              className="input"
            >
              <option value="">Select a dataset</option>
              {datasets.map((dataset) => (
                <option key={dataset.id} value={dataset.id}>
                  {dataset.name} ({dataset.dataset_type})
                </option>
              ))}
            </select>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Number of Clients</label>
              <input
                type="number"
                min="1"
                max="20"
                value={formData.num_clients}
                onChange={(e) => setFormData({ ...formData, num_clients: parseInt(e.target.value) })}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Training Rounds</label>
              <input
                type="number"
                min="1"
                max="1000"
                value={formData.num_rounds}
                onChange={(e) => setFormData({ ...formData, num_rounds: parseInt(e.target.value) })}
                className="input"
              />
            </div>
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Batch Size</label>
              <input
                type="number"
                min="1"
                value={formData.batch_size}
                onChange={(e) => setFormData({ ...formData, batch_size: parseInt(e.target.value) })}
                className="input"
              />
            </div>
            <div>
              <label className="block text-sm font-medium text-gray-700 mb-1">Learning Rate</label>
              <input
                type="number"
                step="0.0001"
                min="0.0001"
                max="0.1"
                value={formData.learning_rate}
                onChange={(e) => setFormData({ ...formData, learning_rate: parseFloat(e.target.value) })}
                className="input"
              />
            </div>
          </div>

          <div className="flex space-x-4">
            <button type="submit" className="flex-1 btn btn-primary">Create Training Job</button>
            <button type="button" onClick={() => navigate('/training')} className="btn btn-secondary">Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );
}
