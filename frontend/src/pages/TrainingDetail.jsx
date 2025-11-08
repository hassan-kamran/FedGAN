import React, { useState, useEffect } from 'react';
import { useParams, Link } from 'react-router-dom';
import { trainingAPI } from '../services/api';

export default function TrainingDetail() {
  const { id } = useParams();
  const [job, setJob] = useState(null);
  const [progress, setProgress] = useState(null);

  useEffect(() => {
    loadJob();
    const interval = setInterval(loadProgress, 5000);
    return () => clearInterval(interval);
  }, [id]);

  const loadJob = async () => {
    try {
      const response = await trainingAPI.get(id);
      setJob(response.data);
    } catch (error) {
      console.error('Error loading job:', error);
    }
  };

  const loadProgress = async () => {
    try {
      const response = await trainingAPI.getProgress(id);
      setProgress(response.data);
    } catch (error) {
      console.error('Error loading progress:', error);
    }
  };

  const handleCancel = async () => {
    if (!window.confirm('Are you sure you want to cancel this training job?')) return;
    try {
      await trainingAPI.cancel(id);
      loadJob();
    } catch (error) {
      alert('Failed to cancel job');
    }
  };

  if (!job) return <div className="flex justify-center"><div className="animate-spin rounded-full h-12 w-12 border-b-2 border-primary-600"></div></div>;

  return (
    <div className="space-y-6">
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold">{job.name}</h2>
          <p className="text-gray-600">{job.description}</p>
        </div>
        <div className="flex space-x-2">
          {job.status === 'running' && (
            <button onClick={handleCancel} className="btn btn-danger">Cancel Job</button>
          )}
          <Link to={`/training/${id}/images`} className="btn btn-secondary">View Images</Link>
          <Link to={`/training/${id}/privacy`} className="btn btn-secondary">Privacy Metrics</Link>
        </div>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="card">
          <p className="text-sm text-gray-600">Status</p>
          <p className="text-2xl font-bold capitalize mt-1">{job.status}</p>
        </div>
        <div className="card">
          <p className="text-sm text-gray-600">Progress</p>
          <p className="text-2xl font-bold mt-1">{job.progress_percentage.toFixed(1)}%</p>
        </div>
        <div className="card">
          <p className="text-sm text-gray-600">Round</p>
          <p className="text-2xl font-bold mt-1">{job.current_round} / {job.num_rounds}</p>
        </div>
      </div>

      <div className="card">
        <h3 className="text-lg font-semibold mb-4">Configuration</h3>
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-sm text-gray-600">Number of Clients</p>
            <p className="font-medium">{job.num_clients}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Batch Size</p>
            <p className="font-medium">{job.batch_size}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Learning Rate</p>
            <p className="font-medium">{job.learning_rate}</p>
          </div>
          <div>
            <p className="text-sm text-gray-600">Latent Dimension</p>
            <p className="font-medium">{job.latent_dim}</p>
          </div>
        </div>
      </div>

      {job.status === 'completed' && (
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Results</h3>
          <div className="grid grid-cols-2 gap-4">
            <div>
              <p className="text-sm text-gray-600">FID Score</p>
              <p className="text-xl font-bold text-primary-600">{job.final_fid_score?.toFixed(2) || 'N/A'}</p>
            </div>
            <div>
              <p className="text-sm text-gray-600">Inception Score</p>
              <p className="text-xl font-bold text-medical-600">{job.final_inception_score?.toFixed(2) || 'N/A'}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
